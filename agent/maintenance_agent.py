"""
Maintenance Agent - LangChain-based agentic AI for predictive maintenance
Handles tool selection, reasoning, and multi-step workflows
"""

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain_core.callbacks.stdout import StdOutCallbackHandler
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
import os
import json
from dotenv import load_dotenv

load_dotenv()


def _safe_stdout_on_chain_start(self, serialized, inputs, **kwargs):
    """Guard LangChain's stdout callback against missing serialized metadata."""
    serialized = serialized or {}
    class_name = serialized.get("name") or (serialized.get("id", [None])[-1]) or "AgentExecutor"
    print(f"\n\n\033[1m> Entering new {class_name} chain...\033[0m")  # noqa: T201


StdOutCallbackHandler.on_chain_start = _safe_stdout_on_chain_start

# ============================================================
# FAILURE MODE THRESHOLDS 
# ============================================================
THRESHOLDS = {
    # Tool Wear Failure: tool fails between 200-240 min wear
    "TWF_WEAR_MIN":     200,
    "TWF_WEAR_MAX":     240,

    # Heat Dissipation Failure: BOTH conditions must be true simultaneously
    "HDF_TEMP_DIFF":    8.6,    # process_temp - air_temp must be < 8.6 K
    "HDF_RPM_MAX":      1380,   # RPM must be below 1380

    # Power Failure: power = torque x (rpm x pi/30)
    "PWF_POWER_MIN":    3500,   # Watts
    "PWF_POWER_MAX":    9000,   # Watts

    # Overstrain Failure: wear x torque product limits per product type
    "OSF_LIMIT_L":      11000,  # minNm for L-type
    "OSF_LIMIT_M":      12000,  # minNm for M-type
    "OSF_LIMIT_H":      13000,  # minNm for H-type

    # Random Failure probability per operation
    "RNF_PROBABILITY":  0.001,

    # Fallback maintenance thresholds (used only if quantiles cannot be computed)
    "MAINTENANCE_URGENT":  200,
    "MAINTENANCE_HIGH":    150,
}


class MaintenanceAgent:
    """
    LangChain-based maintenance agent that uses ReAct prompting to reason about
    machine failures and recommend maintenance actions.
    """

    def __init__(self, df: pd.DataFrame, api_key: Optional[str] = None,
                 provider: str = "Groq (llama3.3-70b)"):
        """
        Initialize the Maintenance Agent

        Args:
            df: Milling machine dataset
            api_key: API key for LLM provider (optional, can be read from .env)
            provider: LLM provider (Groq, Google Gemini, or DeepSeek)
        """
        self.df = df
        self._viz_storage = {}  # Store visualization data indexed by execution step
        self._current_tool_name = None  # Track which tool is currently executing

        if "Groq" in provider:
            self.api_key = os.getenv("GROQ_API_KEY")
        elif "Gemini" in provider:
            self.api_key = os.getenv("GOOGLE_API_KEY")
        elif "DeepSeek" in provider:
            # DeepSeek is routed through OpenRouter in this app.
            self.api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        else:
            self.api_key = api_key

        self.provider = provider
        self.llm = self._initialize_llm()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="input",       # tells memory which key is the user input
            output_key="output",     # tells memory which key is the agent response
            return_messages=False    # return as plain text string, not message objects
        )

        # Initialize ML model for predictions
        from models.predictor import PredictiveModel
        self.predictor = PredictiveModel()
        try:
            # Train ALL models and automatically select the best one
            # Results will be printed to terminal with verbose=True
            training_results = self.predictor.train_all(
                df.copy(), 
                selected_models=None,  # None means train all available models
                use_engineered=False,
                verbose=True  # Enable terminal output for model evaluation
            )
            self.predictor_ready = True
            print(f"\n✓ Model Training Complete! Using '{self.predictor.best_model_name}' for predictions.\n")
        except Exception as e:
            print(f"Failed to train predictor models: {e}")
            self.predictor_ready = False

        # Calculate dynamic limits and domain knowledge from dataset
        self.operating_limits = self._calculate_operating_limits()
        self.maintenance_thresholds = self._calculate_maintenance_thresholds()
        self.domain_knowledge = self._generate_domain_knowledge()

        self.tools = self._create_tools()
        self.agent_executor = self._create_agent()

    def _calculate_operating_limits(self) -> Dict[str, tuple]:
        """
        Calculate operating limits dynamically from dataset min/max values.
        Returns: Dictionary with sensor name -> (min, max) tuple
        """
        sensor_columns = {
            'Rotational speed [rpm]': 'Rotational speed [rpm]',
            'Torque [Nm]': 'Torque [Nm]',
            'Tool wear [min]': 'Tool wear [min]',
            'Air temperature [K]': 'Air temperature [K]',
            'Process temperature [K]': 'Process temperature [K]',
        }
        
        limits = {}
        for display_name, col_name in sensor_columns.items():
            if col_name in self.df.columns:
                min_val = float(self.df[col_name].min())
                max_val = float(self.df[col_name].max())
                limits[display_name] = (round(min_val, 2), round(max_val, 2))
        
        return limits

    def _calculate_maintenance_thresholds(self) -> Dict[str, float]:
        """
        Calculate maintenance tiers from tool-wear quantiles.
        - HIGH threshold  : 80th percentile
        - URGENT threshold: 95th percentile
        Falls back to predefined values if data is unavailable.
        """
        wear_col = 'Tool wear [min]'
        fallback_high = float(THRESHOLDS["MAINTENANCE_HIGH"])
        fallback_urgent = float(THRESHOLDS["MAINTENANCE_URGENT"])

        if wear_col not in self.df.columns:
            return {
                "high": fallback_high,
                "urgent": fallback_urgent,
                "method": "fallback",
            }

        wear_values = pd.to_numeric(self.df[wear_col], errors='coerce').dropna()
        if wear_values.empty:
            return {
                "high": fallback_high,
                "urgent": fallback_urgent,
                "method": "fallback",
            }

        high_q = float(wear_values.quantile(0.80))
        urgent_q = float(wear_values.quantile(0.95))

        # Ensure monotonic thresholds even in degenerate distributions
        if urgent_q <= high_q:
            urgent_q = high_q + 1.0

        return {
            "high": round(high_q, 2),
            "urgent": round(urgent_q, 2),
            "method": "quantile",
        }

    def _generate_domain_knowledge(self) -> Dict[str, Any]:
        """
        Generate domain knowledge strings from actual dataset statistics.
        Returns: Dictionary with dataset metrics calculated from data
        """
        total_ops = len(self.df)
        failures = int(self.df['Machine failure'].sum()) if 'Machine failure' in self.df.columns else 0
        failure_rate = (failures / total_ops * 100) if total_ops > 0 else 0
        
        # Calculate failure mode counts from individual failure columns
        twf_count = int(self.df['TWF'].sum()) if 'TWF' in self.df.columns else 0
        hdf_count = int(self.df['HDF'].sum()) if 'HDF' in self.df.columns else 0
        pwf_count = int(self.df['PWF'].sum()) if 'PWF' in self.df.columns else 0
        osf_count = int(self.df['OSF'].sum()) if 'OSF' in self.df.columns else 0
        rnf_count = int(self.df['RNF'].sum()) if 'RNF' in self.df.columns else 0
        
        # Get sensor ranges from limits
        rpm_min, rpm_max = self.operating_limits.get('Rotational speed [rpm]', (0, 0))
        torque_min, torque_max = self.operating_limits.get('Torque [Nm]', (0, 0))
        wear_min, wear_max = self.operating_limits.get('Tool wear [min]', (0, 0))
        air_temp_min, air_temp_max = self.operating_limits.get('Air temperature [K]', (0, 0))
        proc_temp_min, proc_temp_max = self.operating_limits.get('Process temperature [K]', (0, 0))
        
        return {
            'total_operations': total_ops,
            'total_failures': failures,
            'failure_rate': round(failure_rate, 2),
            'failure_modes': {
                'TWF': twf_count,
                'HDF': hdf_count,
                'PWF': pwf_count,
                'OSF': osf_count,
                'RNF': rnf_count,
            },
            'sensor_ranges': {
                'RPM': f"{rpm_min}-{rpm_max}",
                'Torque': f"{torque_min}-{torque_max}",
                'Wear': f"{wear_min}-{wear_max}",
                'Air_temp': f"{air_temp_min}-{air_temp_max}",
                'Process_temp': f"{proc_temp_min}-{proc_temp_max}",
            }
        }

    def _initialize_llm(self) -> Optional[Any]:
        """Initialize the appropriate LLM based on provider"""
        try:
            if "Groq" in self.provider:
                from langchain_groq import ChatGroq
                if not self.api_key:
                    raise ValueError("Groq API key required")
                return ChatGroq(
                    model="llama-3.3-70b-versatile",
                    api_key=self.api_key,
                    temperature=0.7
                )

            elif "Gemini" in self.provider:
                from langchain_google_genai import ChatGoogleGenerativeAI
                if not self.api_key:
                    raise ValueError("Google Gemini API key required")
                return ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    api_key=self.api_key,
                    temperature=0.7
                )

            elif "DeepSeek" in self.provider:
                from langchain_openai import ChatOpenAI
                if not self.api_key:
                    raise ValueError("OpenRouter/DeepSeek API key required")
                return ChatOpenAI(
                    model="deepseek/deepseek-chat",
                    openai_api_key=self.api_key,
                    openai_api_base="https://openrouter.ai/api/v1",
                    temperature=0.7
                )
        except Exception as e:
            print(f"LLM initialization failed: {e}. Falling back to mock mode.")
            return None

    def _create_tools(self) -> List[Tool]:
        """Create the 7 maintenance intelligence tools"""
        tools = [
            Tool(
                name="predict_failure",
                func=self._predict_failure,
                description="Predict machine failure from sensor data. Input must include JSON with RPM, Torque, ToolWear, AirTemp, ProcessTemp, and Type (L/M/H)."
            ),
            Tool(
                name="analyze_patterns",
                func=self._analyze_patterns,
                description="Analyze statistical patterns in failure data to identify trends across all failure modes."
            ),
            Tool(
                name="diagnose_failure",
                func=self._diagnose_failure,
                description="Diagnose which failure modes (TWF, HDF, PWF, OSF, RNF) are active and count potential cases using dataset rules."
            ),
            Tool(
                name="get_maintenance_schedule",
                func=self._get_maintenance_schedule,
                description="Generate a prioritized maintenance schedule for machines based on risk level."
            ),
            Tool(
                name="calculate_kpis",
                func=self._calculate_kpis,
                description="Calculate operational KPIs including failure rate, MTBF, and availability."
            ),
            Tool(
                name="statistical_summary",
                func=self._statistical_summary,
                description="Get statistical summary of all sensor readings (mean, std, min, max)."
            ),
            Tool(
                name="check_operating_limits",
                func=self._check_operating_limits,
                description="Validate sensor readings against safe operating limits. Input: JSON with sensor values e.g. {RPM: 1500, Torque: 45}."
            ),
        ]
        return tools

    def _first_present_value(self, payload: Dict[str, Any], aliases: List[str]) -> Any:
        """Return the first present key value from alias list, otherwise None."""
        for alias in aliases:
            if alias in payload:
                return payload.get(alias)
        return None

    def _normalize_machine_type(self, raw_value: Any) -> str:
        """Normalize machine type tokens to one of L/M/H."""
        if raw_value is None:
            return ""

        value = str(raw_value).strip().upper().replace("-", "").replace("_", "")
        mapping = {
            "L": "L",
            "M": "M",
            "H": "H",
            "LARGE": "L",
            "MEDIUM": "M",
            "HIGH": "H",
            "LTYPE": "L",
            "MTYPE": "M",
            "HTYPE": "H",
        }
        return mapping.get(value, "")

    def _predict_failure(self, query: str) -> str:
        """Tool 1: Predict failure from sensor data using ML model"""
        try:
            try:
                if isinstance(query, str):
                    sensor_data = json.loads(query.replace("'", '"'))
                else:
                    sensor_data = query
            except json.JSONDecodeError:
                return 'Error: Please provide sensor data in JSON format. Example: {"RPM": 1500, "Torque": 45, "ToolWear": 180, "AirTemp": 298, "ProcessTemp": 308, "Type": "M"}'

            if not self.predictor_ready:
                return "Error: ML model not trained yet. Please ensure training data is available."

            # Define required sensor fields (user must provide ALL for accurate predictions)
            required_fields = {
                'RPM': ['RPM', 'Rotational speed [rpm]'],
                'Torque': ['Torque', 'Torque [Nm]'],
                'Wear': ['Wear', 'ToolWear', 'Tool wear [min]'],
                'AirTemp': ['AirTemp', 'Air temperature [K]'],
                'ProcessTemp': ['ProcessTemp', 'Process temperature [K]'],
                'Type': ['Type', 'type', 'MachineType', 'machine_type', 'ProductType', 'product_type']
            }

            # Check if all required fields are provided by user
            missing_fields = []
            for field_name, aliases in required_fields.items():
                if not any(alias in sensor_data for alias in aliases):
                    missing_fields.append(field_name)

            # MUST have all 6 sensor values for accurate prediction
            if missing_fields:
                missing_str = ", ".join(missing_fields)
                return (
                    f"❌ INCOMPLETE INPUT - Missing required sensor values: {missing_str}\n\n"
                    f"All 6 sensor features MUST be provided for accurate failure prediction:\n"
                    f"  • RPM (Rotational speed in revolutions per minute)\n"
                    f"  • Torque (in Nm - Newton-meters)\n"
                    f"  • Wear (Tool wear in minutes)\n"
                    f"  • AirTemp (Air temperature in Kelvin)\n"
                    f"  • ProcessTemp (Process temperature in Kelvin)\n"
                    f"  • Type (Machine variant: L=Large, M=Medium, or H=High)\n\n"
                    f"Why Type matters:\n"
                    f"  - L-type: OSF threshold = 11,000 minNm\n"
                    f"  - M-type: OSF threshold = 12,000 minNm\n"
                    f"  - H-type: OSF threshold = 13,000 minNm\n"
                    f"  → Different machine types have different overstrain limits!\n\n"
                    f"Example with all values:\n"
                    f'{{"RPM": 1500, "Torque": 45, "ToolWear": 220, "AirTemp": 298, "ProcessTemp": 308, "Type": "M"}}\n\n'
                    f"Current dataset operating ranges:\n"
                    f"  • RPM: {self.operating_limits.get('Rotational speed [rpm]', (0, 0))}\n"
                    f"  • Torque: {self.operating_limits.get('Torque [Nm]', (0, 0))} Nm\n"
                    f"  • Wear: {self.operating_limits.get('Tool wear [min]', (0, 0))} min\n"
                    f"  • Air Temp: {self.operating_limits.get('Air temperature [K]', (0, 0))} K\n"
                    f"  • Process Temp: {self.operating_limits.get('Process temperature [K]', (0, 0))} K"
                )

            # All values provided - extract features
            features = {}
            features['Air temperature [K]']      = sensor_data.get('AirTemp',     sensor_data.get('Air temperature [K]'))
            features['Process temperature [K]']  = sensor_data.get('ProcessTemp', sensor_data.get('Process temperature [K]'))
            features['Rotational speed [rpm]']   = sensor_data.get('RPM',         sensor_data.get('Rotational speed [rpm]'))
            features['Torque [Nm]']              = sensor_data.get('Torque',       sensor_data.get('Torque [Nm]'))
            features['Tool wear [min]']          = sensor_data.get('Wear',         sensor_data.get('ToolWear', sensor_data.get('Tool wear [min]')))

            # Validate numeric inputs before model inference.
            numeric_errors = []
            for key in [
                'Air temperature [K]',
                'Process temperature [K]',
                'Rotational speed [rpm]',
                'Torque [Nm]',
                'Tool wear [min]'
            ]:
                try:
                    features[key] = float(features[key])
                except (TypeError, ValueError):
                    numeric_errors.append(key)

            if numeric_errors:
                return (
                    "❌ INVALID INPUT - These fields must be numeric values: "
                    + ", ".join(numeric_errors)
                )

            # Build sample DataFrame with one-hot encoded Type
            sample_df = pd.DataFrame([features])
            raw_machine_type = self._first_present_value(sensor_data, required_fields['Type'])
            machine_type = self._normalize_machine_type(raw_machine_type)
            if machine_type not in {'L', 'M', 'H'}:
                return "❌ INVALID INPUT - Type must be one of: L, M, H"

            sample_df['Type'] = machine_type
            sample_df = pd.get_dummies(sample_df, columns=['Type'])

            # Align columns to training feature set
            for col in self.predictor.feature_names:
                if col not in sample_df.columns:
                    sample_df[col] = 0

            X_sample = sample_df[self.predictor.feature_names]

            result = self.predictor.predict(X_sample)
            if result is None:
                return "Error: Model prediction failed. Model may not be trained."

            failure_risk = float(result['probability'][0][1] * 100.0)
            prediction = int(result['prediction'][0])
            status = "HIGH RISK" if failure_risk > 70 else "MEDIUM RISK" if failure_risk > 40 else "LOW RISK"

            # Reliability checks for out-of-distribution inputs.
            raw_input = dict(features)
            raw_input['Type'] = machine_type
            quality = self.predictor.assess_input_quality(raw_input)

            # Model certainty from predicted class probability (0..1).
            pred_prob = failure_risk / 100.0
            class_certainty = pred_prob if prediction == 1 else (1.0 - pred_prob)
            confidence = 0.7 * class_certainty + 0.3 * quality.get("confidence_component", 0.5)
            confidence = float(np.clip(confidence, 0.0, 1.0))
            confidence_pct = confidence * 100.0
            
            # Get failure mode analysis
            failure_modes = result.get('failure_modes', [])

            # Human-readable reliability notes
            range_warnings = []
            for item in quality.get("out_of_range", []):
                range_warnings.append(
                    f"{item['feature']}={item['value']:.2f} outside [{item['min']:.2f}, {item['max']:.2f}]"
                )

            z_warnings = []
            for item in quality.get("high_z_features", []):
                z_warnings.append(f"{item['feature']} z={item['z_score']:.2f}")

            confidence_label = (
                "HIGH" if confidence_pct >= 80 else
                "MEDIUM" if confidence_pct >= 60 else
                "LOW"
            )

            # Store structured data for visualization
            self._viz_storage["predict_failure"] = {
                "tool":           "predict_failure",
                "failure_risk":   failure_risk,
                "prediction":     prediction,
                "status":         status,
                "confidence":     confidence_pct,
                "confidence_label": confidence_label,
                "failure_modes":  failure_modes,
                "anomaly_detected": quality.get("anomaly_detected", False),
                "anomaly_score": quality.get("anomaly_score", 0.0),
                "range_warnings": range_warnings,
                "z_warnings": z_warnings,
                "features": {
                    "RPM":            features.get('Rotational speed [rpm]'),
                    "Torque (Nm)":    features.get('Torque [Nm]'),
                    "Tool Wear (min)": features.get('Tool wear [min]'),
                    "Air Temp (K)":   features.get('Air temperature [K]'),
                    "Process Temp (K)": features.get('Process temperature [K]'),
                    "Type":           machine_type
                },
                "input_status": {
                    "all_provided": True,
                    "defaults_used": []
                }
            }

            # Build response with failure mode info
            modes_str = ""
            if failure_modes:
                modes_str = "Potential Failure Modes: " + ", ".join([f"{m['mode']}({m['risk']:.0f}%)" for m in failure_modes])

            reliability_note = ""
            if quality.get("anomaly_detected"):
                reliability_note = (
                    " | Reliability Warning: input appears out-of-distribution "
                    "(outside training range and/or high Z-score)"
                )
            
            return (
                f"Prediction: {status} | Failure Risk: {failure_risk:.1f}% | "
                f"Confidence: {confidence_pct:.1f}% ({confidence_label}) | "
                f"Outcome: {'⚠️ FAILURE LIKELY' if prediction == 1 else '✓ No Failure Expected'} | "
                f"{modes_str}{reliability_note}"
            )
        except Exception as e:
            return f"Error in prediction: {str(e)}"

    def _analyze_patterns(self, query: str) -> str:
        """Tool 2: Analyze failure patterns across all modes"""
        try:
            failure_modes = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
            pattern_summary = {}
            for mode in failure_modes:
                if mode in self.df.columns:
                    pattern_summary[mode] = int(self.df[mode].sum())

            # Store structured data for visualization
            self._viz_storage["analyze_patterns"] = {
                "tool":     "analyze_patterns",
                "patterns": pattern_summary
            }

            modes_list = ', '.join([f"{mode}({count})" for mode, count in sorted(pattern_summary.items(), key=lambda x: x[1], reverse=True)])
            return f"Failure Mode Distribution: {modes_list}"
        except Exception as e:
            return f"Error in pattern analysis: {str(e)}"

    def _diagnose_failure(self, query: str) -> str:
        """Tool 3: Diagnose failure modes - read actual failure mode indicators from dataset"""
        try:
            # Read actual failure mode counts directly from dataset
            # These are binary indicators (0 or 1) that show whether each failure mode occurred
            twf_count = int(self.df['TWF'].sum()) if 'TWF' in self.df.columns else 0
            hdf_count = int(self.df['HDF'].sum()) if 'HDF' in self.df.columns else 0
            pwf_count = int(self.df['PWF'].sum()) if 'PWF' in self.df.columns else 0
            osf_count = int(self.df['OSF'].sum()) if 'OSF' in self.df.columns else 0
            rnf_count = int(self.df['RNF'].sum()) if 'RNF' in self.df.columns else 0

            # Store structured data for visualization
            self._viz_storage["diagnose_failure"] = {
                "tool": "diagnose_failure",
                "diagnosis": {
                    "Tool Wear (TWF)":         twf_count,
                    "Heat Dissipation (HDF)":  hdf_count,
                    "Power Failure (PWF)":     pwf_count,
                    "Overstrain (OSF)":        osf_count,
                    "Random Failure (RNF)":    rnf_count,
                }
            }

            return f"Failure Diagnosis: TWF={twf_count}, HDF={hdf_count}, PWF={pwf_count}, OSF={osf_count}, RNF={rnf_count}"
        except Exception as e:
            return f"Error in diagnosis: {str(e)}"

    def _get_maintenance_schedule(self, query: str) -> str:
        """Tool 4: Generate prioritized maintenance schedule"""
        try:
            wear_col = 'Tool wear [min]'
            high_threshold = float(self.maintenance_thresholds["high"])
            urgent_threshold = float(self.maintenance_thresholds["urgent"])
            threshold_method = self.maintenance_thresholds.get("method", "quantile")

            urgent_units = self.df[self.df[wear_col] >= urgent_threshold]
            high_units = self.df[
                (self.df[wear_col] >= high_threshold) &
                (self.df[wear_col] < urgent_threshold)
            ]
            routine_units = self.df[self.df[wear_col] < high_threshold]

            # Store structured data for visualization
            self._viz_storage["get_maintenance_schedule"] = {
                "tool": "get_maintenance_schedule",
                "thresholds": {
                    "method": threshold_method,
                    "q80_high": round(high_threshold, 2),
                    "q95_urgent": round(urgent_threshold, 2),
                },
                "schedule": [
                    {
                        "Priority":  "🔴 URGENT",
                        "Condition": f"Tool wear >= {urgent_threshold:.2f} min ({'Q95' if threshold_method == 'quantile' else 'fallback'})",
                        "Units":     len(urgent_units),
                        "Action":    "Replace tool immediately — TWF imminent"
                    },
                    {
                        "Priority":  "🟡 HIGH",
                        "Condition": f"Tool wear >= {high_threshold:.2f} and < {urgent_threshold:.2f} min ({'Q80-Q95' if threshold_method == 'quantile' else 'fallback range'})",
                        "Units":     len(high_units),
                        "Action":    "Schedule accelerated inspection"
                    },
                    {
                        "Priority":  "🟢 ROUTINE",
                        "Condition": f"Tool wear < {high_threshold:.2f} min",
                        "Units":     len(routine_units),
                        "Action":    "Schedule within next maintenance window"
                    },
                ]
            }

            return (
                f"Maintenance Schedule ({threshold_method} thresholds: "
                f"Q80={high_threshold:.2f} min, Q95={urgent_threshold:.2f} min): "
                f"URGENT={len(urgent_units)} units, "
                f"HIGH={len(high_units)} units, "
                f"ROUTINE={len(routine_units)} units"
            )
        except Exception as e:
            return f"Error in scheduling: {str(e)}"

    def _calculate_kpis(self, query: str) -> str:
        """Tool 5: Calculate operational KPIs"""
        try:
            total_records = len(self.df)
            failures      = int(self.df['Machine failure'].sum())
            failure_rate  = (failures / total_records) * 100
            mtbf          = total_records / max(failures, 1)
            availability  = 100 - failure_rate

            # Store structured data for visualization
            self._viz_storage["calculate_kpis"] = {
                "tool": "calculate_kpis",
                "kpis": {
                    "Total Operations": total_records,
                    "Failures":         failures,
                    "Failure Rate":     f"{failure_rate:.2f}%",
                    "MTBF":             f"{mtbf:.0f} ops",
                    "Availability":     f"{availability:.2f}%",
                }
            }

            return f"KPIs: Total={total_records}, Failures={failures}, Rate={failure_rate:.2f}%, MTBF={mtbf:.0f}ops, Availability={availability:.2f}%"
        except Exception as e:
            return f"Error in KPI calculation: {str(e)}"

    def _statistical_summary(self, query: str) -> str:
        """Tool 6: Statistical summary of all sensor readings"""
        try:
            numeric_cols = [
                'Air temperature [K]', 'Process temperature [K]',
                'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
            ]
            stats   = {}
            summary = "Statistical Summary:\n"

            for col in numeric_cols:
                if col in self.df.columns:
                    mean = self.df[col].mean()
                    std  = self.df[col].std()
                    stats[col] = {
                        "mean": round(mean, 2),
                        "std":  round(std, 2),
                        "min":  round(self.df[col].min(), 2),
                        "max":  round(self.df[col].max(), 2),
                    }
                    summary += f"• {col}: μ={mean:.1f}, σ={std:.1f}\n"

            # Store structured data for visualization
            self._viz_storage["statistical_summary"] = {
                "tool": "statistical_summary",
                "statistics": stats
            }

            stats_summary = "; ".join([f"{col}({stat['mean']:.1f}±{stat['std']:.1f})" for col, stat in stats.items()])
            return f"Statistical Summary: {stats_summary}"
        except Exception as e:
            return f"Error in statistical summary: {str(e)}"

    def _check_operating_limits(self, query: str) -> str:
        """Tool 7: Validate sensor readings against safe operating limits"""
        # Operating limits calculated dynamically from dataset
        limits = self.operating_limits

        # Map short input keys to full column names
        key_map = {
            "RPM":         "Rotational speed [rpm]",
            "Torque":      "Torque [Nm]",
            "ToolWear":    "Tool wear [min]",
            "Wear":        "Tool wear [min]",
            "AirTemp":     "Air temperature [K]",
            "ProcessTemp": "Process temperature [K]",
        }

        try:
            sensor_data = json.loads(query.replace("'", '"'))
        except Exception:
            sensor_data = {}

        result     = "Operating Limit Validation:\n"
        limit_rows = []

        if sensor_data:
            # Check only the provided sensor values against their limits
            for short_key, value in sensor_data.items():
                col = key_map.get(short_key, short_key)
                if col in limits:
                    lo, hi  = limits[col]
                    within  = lo <= float(value) <= hi
                    status  = "✓ OK" if within else f"⚠ OUT OF RANGE ({lo}–{hi})"
                    result += f"• {col}: {value} → {status}\n"
                    limit_rows.append({
                        "Sensor":  col,
                        "Value":   value,
                        "Min":     lo,
                        "Max":     hi,
                        "Status":  "✓ OK" if within else "⚠ OUT OF RANGE"
                    })
        else:
            # Fallback: summarise dataset-wide violations
            for col, (lo, hi) in limits.items():
                if col in self.df.columns:
                    violations = len(self.df[(self.df[col] < lo) | (self.df[col] > hi)])
                    status     = "✓ OK" if violations == 0 else f"⚠ {violations} violations"
                    result    += f"• {col}: {status}\n"
                    limit_rows.append({
                        "Sensor":     col,
                        "Min":        lo,
                        "Max":        hi,
                        "Violations": violations,
                        "Status":     status
                    })

        # Store structured data for visualization
        self._viz_storage["check_operating_limits"] = {
            "tool":   "check_operating_limits",
            "limits": limit_rows
        }
        
        summary = f"Operating Limits Check: {len(limit_rows)} sensors validated" if limit_rows else "Operating Limits: No data provided"
        return summary

    def _create_agent(self) -> Optional[AgentExecutor]:
        """Create LangChain ReAct agent executor"""
        if self.llm is None:
            return None

        # Build dynamic domain knowledge string from dataset statistics
        dk = self.domain_knowledge
        failure_breakdown = ', '.join([f"{k}({v})" for k, v in dk['failure_modes'].items()])
        
        # Create prompt with proper input_variables for ReAct agent
        prompt = PromptTemplate(
            input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
            template=f"""You are an Advanced Reasoning Intelligence Agent, an expert AI system for predictive maintenance of industrial milling machines at Anywhere.

IMPORTANT: After every tool call, the UI will AUTOMATICALLY generate charts and tables from your tool results. You do NOT need to generate charts yourself — just call the right tool and the visualization will appear below your answer.

DOMAIN KNOWLEDGE (Calculated from live dataset):
- Dataset: {dk['total_operations']:,} milling machine operations, {dk['failure_rate']}% failure rate, {dk['total_failures']} total failures
- Failure mode counts: {failure_breakdown}
- Failure mode rules: TWF (wear>200min), HDF (temp_diff<8.6K AND rpm<1380), PWF (power<3500W or >9000W), OSF (wear×torque exceeds L:11000 M:12000 H:13000), RNF (0.1% random)
- Normal sensor ranges: RPM {dk['sensor_ranges']['RPM']}, Torque {dk['sensor_ranges']['Torque']}Nm, Wear {dk['sensor_ranges']['Wear']}min, Air temp {dk['sensor_ranges']['Air_temp']}K, Process temp {dk['sensor_ranges']['Process_temp']}K

RULES:
- Greetings or off-topic questions: answer directly WITHOUT tools
- Any maintenance/sensor/failure/KPI/schedule question: use the appropriate tool
- ALWAYS end maintenance answers with a Risk Assessment and Recommended Action
- NEVER say you cannot display charts — the UI handles visualization automatically

Tools available:
{{tools}}

Valid tool names: [{{tool_names}}]

FORMAT WHEN TOOL IS NEEDED:
Thought: [reasoning about which tool to use]
Action: [tool name]
Action Input: [tool input]
Observation: [tool result]
Thought: I now know the final answer
Final Answer:
##### Analysis
- [Write clear bullet points explaining the result]
##### Risk Assessment: 
- [🔴/🟡/🟢 + explanation]
##### Recommended Action: 
- [specific next step]

FORMAT WHEN NO TOOL NEEDED:
Thought: This is a conversational question, no tool needed
Final Answer: [helpful response]

Question: {{input}}
Thought:{{agent_scratchpad}}"""
        )

        agent    = create_react_agent(self.llm, self.tools, prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=8,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
        return executor

    def run(self, query: str) -> Dict[str, Any]:
        """
        Run the agent with a user query

        Args:
            query: User question about maintenance

        Returns:
            Dictionary with answer, tool calls, and visualization data
        """
        if self.agent_executor is None:
            return self._run_mock_mode(query)

        try:
            result = self.agent_executor.invoke({"input": query})

            tool_calls = []
            if "intermediate_steps" in result:
                for step_idx, step in enumerate(result["intermediate_steps"]):
                    tool_name = step[0].tool
                    # Retrieve visualization data stored during tool execution
                    viz_key = f"{tool_name}_{step_idx}"
                    visualization = self._viz_storage.get(viz_key, self._viz_storage.get(tool_name, {}))
                    
                    tool_calls.append({
                        "tool":          tool_name,
                        "input":         step[0].tool_input,
                        "output":        step[1],
                        "visualization": visualization
                    })

            return {
                "answer":     result.get("output", "Unable to process query"),
                "tool_calls": tool_calls,
                "success":    True,
                "mode":       "LLM"
            }
        except Exception as e:
            print(f"Agent execution error: {e}")
            return self._run_mock_mode(query)
        finally:
            # Clear visualization storage after use to prevent stale data
            self._viz_storage = {}

    def _run_mock_mode(self, query: str) -> Dict[str, Any]:
        """Fallback mock agent when LLM is unavailable — uses keyword routing"""
        query_lower = query.lower()
        tool_used   = None

        if "predict" in query_lower or "failure" in query_lower:
            answer    = self._predict_failure(query)
            tool_used = "predict_failure"
        elif "pattern" in query_lower or "analyze" in query_lower:
            answer    = self._analyze_patterns(query)
            tool_used = "analyze_patterns"
        elif "diagnose" in query_lower:
            answer    = self._diagnose_failure(query)
            tool_used = "diagnose_failure"
        elif "schedule" in query_lower or "maintenance" in query_lower:
            answer    = self._get_maintenance_schedule(query)
            tool_used = "get_maintenance_schedule"
        elif "kpi" in query_lower or "performance" in query_lower:
            answer    = self._calculate_kpis(query)
            tool_used = "calculate_kpis"
        elif "statistic" in query_lower or "summary" in query_lower:
            answer    = self._statistical_summary(query)
            tool_used = "statistical_summary"
        elif "limit" in query_lower or "threshold" in query_lower:
            answer    = self._check_operating_limits(query)
            tool_used = "check_operating_limits"
        else:
            answer = (
                "🤖 **I'm in Mock Mode** (no LLM API configured)\n\n"
                "I can help with:\n"
                "• 🔮 Failure prediction from sensor data\n"
                "• 📊 Pattern analysis and trend detection\n"
                "• 🔍 Failure diagnosis and root cause\n"
                "• 📋 Maintenance schedule generation\n"
                "• 📈 KPI and performance metrics\n"
                "• 🧮 Statistical analysis\n"
                "• ⚙️ Operating limit validation\n\n"
                '**Try asking:** "Predict failure for RPM=1500, Torque=45, Wear=180"'
            )
            tool_used = None

        tool_calls = []
        if tool_used:
            tool_calls.append({
                "tool":          tool_used,
                "input":         query,
                "output":        answer,
                "visualization": self._viz_storage.get(tool_used, {})
            })

        result = {
            "answer":     answer,
            "tool_calls": tool_calls,
            "success":    True,
            "mode":       "MOCK"
        }
        
        self._viz_storage = {}
        
        return result