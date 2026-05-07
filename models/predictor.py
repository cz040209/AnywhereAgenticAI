"""
Predictive Model - ML Models for failure prediction
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


class PredictiveModel:
    """
    ML Model trainer and predictor for machine failure
    
    IMPORTANT: This model uses ONLY sensor features for prediction, NOT failure mode indicators.
    This avoids data leakage and creates realistic predictions.
    
    Sensor features used:
    - Air temperature [K]
    - Process temperature [K]
    - Rotational speed [rpm]
    - Torque [Nm]
    - Tool wear [min]
    - Type (machine type: L, M, H)
    
    Features EXCLUDED (to avoid data leakage):
    - TWF, HDF, PWF, OSF, RNF (these are derived from failure logic, not predictive features)
    
    Target: Machine failure (1 = failure, 0 = no failure)
    """

    def __init__(self):
        self.models = {}
        self.results = {}
        self.scaler = None
        self.best_model_name = None  # Track the best performing model
        self.feature_names = None
        self.failure_mode_model = None  # Optional: model to predict which failure mode
        self.risk_calibration = {}  # Empirical risk calibration built from dataset
        self.feature_bounds = {}  # Dataset min/max bounds per numeric sensor feature
        self.feature_stats = {}  # Dataset mean/std per numeric sensor feature

    def train_all(
        self,
        df,
        test_size=0.2,
        use_smote=True,
        scale_features=True,
        use_engineered=False,
        selected_models=None,
        verbose=True
    ):
        """
        Train multiple ML models and select the best one
        
        Args:
            df: Input DataFrame with features and target
            test_size: Test split ratio
            use_smote: Whether to use SMOTE (currently not implemented)
            scale_features: Whether to scale features
            use_engineered: Whether to use engineered features
            selected_models: List of models to train. If None, trains all models
            verbose: Whether to print results to terminal
            
        Returns:
            Dictionary with evaluation results for all trained models
        """

        # Default to training all models if none specified
        if selected_models is None:
            selected_models = [
                "Random Forest",
                "Gradient Boosting",
                "Logistic Regression",
                "Decision Tree",
                "Neural Network MLP"
            ]

        if verbose:
            print("\n" + "="*80)
            print("MODEL TRAINING AND EVALUATION PIPELINE")
            print("="*80)
            print(f"Training {len(selected_models)} models: {', '.join(selected_models)}")
            print(f"Dataset size: {len(df)} samples")
            print(f"Test split: {test_size*100:.0f}%, Train split: {(1-test_size)*100:.0f}%")
            print("="*80)
            print("\n⚠️  DATA PREPROCESSING: Removing failure mode indicators to avoid data leakage")
            print("   - Removed columns: TWF, HDF, PWF, OSF, RNF")
            print("   - Only sensor features used: Air Temp, Process Temp, RPM, Torque, Wear, Type")
            print("="*80 + "\n")

        # Build empirical calibration curves for failure mode risk scoring.
        # This keeps risk outputs grounded in observed dataset distributions.
        self.risk_calibration = self._build_risk_calibration(df)
        self._build_feature_statistics(df)

        # Remove UDI and Product ID (not useful for prediction)
        df = df.drop(columns=['UDI', 'Product ID'])
        
        # CRITICAL: Remove failure mode indicator columns to avoid data leakage
        # These columns are derived from the failure logic, not independent features
        failure_mode_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        df = df.drop(columns=[col for col in failure_mode_cols if col in df.columns])

        # Encode categorical column (Type: L, M, H)
        df = pd.get_dummies(df, columns=["Type"])

        # define x and y after encoding
        X = df.drop(columns=['Machine failure']).fillna(0)
        y = df['Machine failure'].fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )

        if verbose:
            print(f"Class distribution:")
            print(f"  Training set: {(y_train==0).sum()} negative, {(y_train==1).sum()} positive")
            print(f"  Test set: {(y_test==0).sum()} negative, {(y_test==1).sum()} positive")
            print()

        # store feature names for later prediction
        self.feature_names = list(X.columns) 

        # Feature scaling (for LR + MLP and all models)
        if scale_features:
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train.values)
            X_test_scaled = self.scaler.transform(X_test.values)
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values

        results = {}

        # Train Random Forest
        if "Random Forest" in selected_models:
            if verbose:
                print("Training: Random Forest Classifier...")
            rf = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train_scaled, y_train)
            y_pred = rf.predict(X_test_scaled)
            y_proba = rf.predict_proba(X_test_scaled)[:, 1]
            
            results["Random Forest"] = self._evaluate(y_test, y_pred, y_proba)
            self.models["Random Forest"] = rf
            if verbose:
                self._print_results("Random Forest", results["Random Forest"])

        # Train Gradient Boosting
        if "Gradient Boosting" in selected_models:
            if verbose:
                print("Training: Gradient Boosting Classifier...")
            gb = GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            )
            gb.fit(X_train_scaled, y_train)
            y_pred = gb.predict(X_test_scaled)
            y_proba = gb.predict_proba(X_test_scaled)[:, 1]
            
            results["Gradient Boosting"] = self._evaluate(y_test, y_pred, y_proba)
            self.models["Gradient Boosting"] = gb
            if verbose:
                self._print_results("Gradient Boosting", results["Gradient Boosting"])

        # Train Logistic Regression
        if "Logistic Regression" in selected_models:
            if verbose:
                print("Training: Logistic Regression...")
            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(X_train_scaled, y_train)
            y_pred = lr.predict(X_test_scaled)
            y_proba = lr.predict_proba(X_test_scaled)[:, 1]
            
            results["Logistic Regression"] = self._evaluate(y_test, y_pred, y_proba)
            self.models["Logistic Regression"] = lr
            if verbose:
                self._print_results("Logistic Regression", results["Logistic Regression"])

        # Train Decision Tree
        if "Decision Tree" in selected_models:
            if verbose:
                print("Training: Decision Tree Classifier...")
            dt = DecisionTreeClassifier(random_state=42, max_depth=10)
            dt.fit(X_train_scaled, y_train)
            y_pred = dt.predict(X_test_scaled)
            y_proba = dt.predict_proba(X_test_scaled)[:, 1]
            
            results["Decision Tree"] = self._evaluate(y_test, y_pred, y_proba)
            self.models["Decision Tree"] = dt
            if verbose:
                self._print_results("Decision Tree", results["Decision Tree"])

        # Train Neural Network MLP
        if "Neural Network MLP" in selected_models:
            if verbose:
                print("Training: Neural Network (MLP)...")
            mlp = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=300,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
            mlp.fit(X_train_scaled, y_train)
            y_pred = mlp.predict(X_test_scaled)
            y_proba = mlp.predict_proba(X_test_scaled)[:, 1]
            
            results["Neural Network MLP"] = self._evaluate(y_test, y_pred, y_proba)
            self.models["Neural Network MLP"] = mlp
            if verbose:
                self._print_results("Neural Network MLP", results["Neural Network MLP"])

        # Select best model based on recall
        self.results = results
        self.best_model_name = self._select_best_model(results, verbose)
        
        if verbose:
            print("\n" + "="*80)
            print(f"✓ BEST MODEL SELECTED: {self.best_model_name}")
            print("="*80)
            print(f"This model will be used for all predictions.\n")
        
        return results

    def _build_risk_calibration(self, df):
        """
        Build empirical distributions used to convert rule violations into risk percentages.
        All scaling values come from observed dataset values (no arbitrary divisors).
        """
        required_cols = {
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
            "Type",
        }

        if not required_cols.issubset(df.columns):
            return {}

        air_temp = pd.to_numeric(df["Air temperature [K]"], errors="coerce")
        process_temp = pd.to_numeric(df["Process temperature [K]"], errors="coerce")
        rpm = pd.to_numeric(df["Rotational speed [rpm]"], errors="coerce")
        torque = pd.to_numeric(df["Torque [Nm]"], errors="coerce")
        wear = pd.to_numeric(df["Tool wear [min]"], errors="coerce")
        machine_type = df["Type"].astype(str)

        temp_diff = process_temp - air_temp
        power = torque * rpm * np.pi / 30.0

        twf_wear_excess = (wear - 200.0)[wear > 200.0].dropna().to_numpy()

        hdf_mask = (temp_diff < 8.6) & (rpm < 1380.0)
        hdf_temp_margin = (8.6 - temp_diff)[hdf_mask].dropna().to_numpy()
        hdf_rpm_margin = (1380.0 - rpm)[hdf_mask].dropna().to_numpy()

        pwf_low_excess = (3500.0 - power)[power < 3500.0].dropna().to_numpy()
        pwf_high_excess = (power - 9000.0)[power > 9000.0].dropna().to_numpy()

        overstrain = wear * torque
        osf_limits = {"L": 11000.0, "M": 12000.0, "H": 13000.0}
        osf_excess_by_type = {}
        for p_type, limit in osf_limits.items():
            mask = (machine_type == p_type) & (overstrain > limit)
            osf_excess_by_type[p_type] = (overstrain - limit)[mask].dropna().to_numpy()

        return {
            "twf_wear_excess": twf_wear_excess,
            "hdf_temp_margin": hdf_temp_margin,
            "hdf_rpm_margin": hdf_rpm_margin,
            "pwf_low_excess": pwf_low_excess,
            "pwf_high_excess": pwf_high_excess,
            "osf_excess_by_type": osf_excess_by_type,
        }

    def _empirical_risk(self, value, distribution):
        """
        Convert a violation magnitude into a 0-100 risk via empirical percentile.
        """
        if distribution is None:
            return 0.0

        values = np.asarray(distribution, dtype=float)
        if values.size == 0:
            return 0.0

        values = np.sort(values)
        rank = np.searchsorted(values, float(value), side="right")
        percentile = (rank / values.size) * 100.0
        return float(np.clip(percentile, 0.0, 100.0))

    def _evaluate(self, y_test, y_pred, y_proba):
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
    
    def _print_results(self, model_name, metrics):
        """Print evaluation metrics for a model in a formatted way"""
        print(f"  Results:")
        print(f"    Accuracy:  {metrics['accuracy']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    F1-Score:  {metrics['f1']:.4f}")
        print(f"    ROC-AUC:   {metrics['roc_auc']:.4f}")
        cm = metrics['confusion_matrix']
        print(f"    Confusion Matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")
        print()
    
    def _select_best_model(self, results, verbose=True):
        """Select best model based on recall (primary) and ROC-AUC (secondary)."""
        if not results:
            return None
        
        # Sort by recall (primary), then by ROC-AUC (secondary)
        best_model_name = max(
            results.keys(),
            key=lambda x: (results[x]['recall'], results[x]['roc_auc'])
        )
        
        if verbose:
            best_recall = results[best_model_name]['recall']
            best_auc = results[best_model_name]['roc_auc']
            print(f"\nModel Comparison (sorted by Recall):")
            print("-" * 80)
            print(f"{'Model':<25} {'Recall':<12} {'ROC-AUC':<12} {'Accuracy':<12} {'Precision':<12} {'F1-Score':<12}")
            print("-" * 80)
            
            for model_name in sorted(results.keys(), key=lambda x: results[x]['recall'], reverse=True):
                metrics = results[model_name]
                print(f"{model_name:<25} {metrics['recall']:<12.4f} {metrics['roc_auc']:<12.4f} "
                      f"{metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} {metrics['f1']:<12.4f}")
            print("-" * 80)
        
        return best_model_name

    def predict(self, features):
        """
        Make prediction using the best trained model
        Also analyzes which failure mode is likely based on sensor parameters
        
        Args:
            features: DataFrame or numpy array of features
            
        Returns:
            Dictionary with predictions, probabilities, and failure mode analysis
        """
        if self.best_model_name is None or self.best_model_name not in self.models:
            return None

        model = self.models[self.best_model_name]

        # Handle both DataFrame and numpy array inputs
        if isinstance(features, pd.DataFrame):
            features_for_scaling = features.values
        else:
            features_for_scaling = features

        if self.scaler is not None:
            features_scaled = self.scaler.transform(features_for_scaling)
        else:
            features_scaled = features_for_scaling

        prediction = model.predict(features_scaled)
        probability = model.predict_proba(features_scaled)

        # Analyze which failure mode is likely (based on sensor thresholds)
        failure_modes = self._analyze_failure_modes(features)
        
        return {
            "prediction": prediction,
            "probability": probability,
            "failure_modes": failure_modes
        }

    def _build_feature_statistics(self, df):
        """Capture training distribution stats for OOD and anomaly checks."""
        numeric_features = [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
        ]

        self.feature_bounds = {}
        self.feature_stats = {}

        for col in numeric_features:
            if col not in df.columns:
                continue

            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if series.empty:
                continue

            mean_val = float(series.mean())
            std_val = float(series.std())
            self.feature_bounds[col] = (float(series.min()), float(series.max()))
            self.feature_stats[col] = {
                "mean": mean_val,
                "std": std_val if std_val > 0 else 1e-9,
            }

    def assess_input_quality(self, features):
        """
        Assess whether inputs are in-distribution using:
        1) hard min/max bounds from training data
        2) per-feature z-scores from training mean/std
        """
        if not self.feature_bounds or not self.feature_stats:
            return {
                "out_of_range": [],
                "z_scores": {},
                "high_z_features": [],
                "max_abs_z": 0.0,
                "anomaly_score": 0.0,
                "anomaly_detected": False,
                "confidence_component": 0.5,
            }

        normalized_inputs = {}
        if isinstance(features, pd.DataFrame) and not features.empty:
            row = features.iloc[0]
            for col in self.feature_bounds.keys():
                if col in features.columns:
                    normalized_inputs[col] = row[col]
        elif isinstance(features, dict):
            alias_map = {
                "Air temperature [K]": ["Air temperature [K]", "AirTemp"],
                "Process temperature [K]": ["Process temperature [K]", "ProcessTemp"],
                "Rotational speed [rpm]": ["Rotational speed [rpm]", "RPM"],
                "Torque [Nm]": ["Torque [Nm]", "Torque"],
                "Tool wear [min]": ["Tool wear [min]", "ToolWear", "Wear"],
            }
            for canonical, aliases in alias_map.items():
                for alias in aliases:
                    if alias in features:
                        normalized_inputs[canonical] = features[alias]
                        break

        out_of_range = []
        z_scores = {}
        high_z_features = []
        max_abs_z = 0.0

        for col, value in normalized_inputs.items():
            if col not in self.feature_bounds or col not in self.feature_stats:
                continue

            try:
                value_f = float(value)
            except (ValueError, TypeError):
                continue

            min_val, max_val = self.feature_bounds[col]
            if value_f < min_val or value_f > max_val:
                out_of_range.append({
                    "feature": col,
                    "value": value_f,
                    "min": min_val,
                    "max": max_val,
                })

            mean_val = self.feature_stats[col]["mean"]
            std_val = self.feature_stats[col]["std"]
            z = (value_f - mean_val) / std_val
            abs_z = abs(float(z))
            z_scores[col] = float(z)
            max_abs_z = max(max_abs_z, abs_z)

            if abs_z >= 3.0:
                high_z_features.append({
                    "feature": col,
                    "value": value_f,
                    "z_score": float(z),
                })

        # 0..1 anomaly score; >=0.5 means strong outlier behavior
        z_score_component = min(max_abs_z / 4.0, 1.0)
        range_component = 1.0 if out_of_range else 0.0
        anomaly_score = 0.6 * z_score_component + 0.4 * range_component

        anomaly_detected = bool(out_of_range or high_z_features)
        confidence_component = 1.0 - anomaly_score
        confidence_component = float(np.clip(confidence_component, 0.0, 1.0))

        return {
            "out_of_range": out_of_range,
            "z_scores": z_scores,
            "high_z_features": high_z_features,
            "max_abs_z": float(max_abs_z),
            "anomaly_score": float(anomaly_score),
            "anomaly_detected": anomaly_detected,
            "confidence_component": confidence_component,
        }
    
    def _analyze_failure_modes(self, features):
        """
        Analyze which failure mode is likely based on sensor parameters
        
        Uses the known failure thresholds from the dataset specification:
        - TWF: Tool wear > 200 min
        - HDF: Temp diff < 8.6K AND RPM < 1380
        - PWF: Power (torque × RPM × π/30) < 3500W or > 9000W
        - OSF: Tool wear × Torque exceeds type limit (L:11000, M:12000, H:13000)
        - RNF: Random (0.1% probability)
        """
        
        # Extract feature values - no fallback defaults; missing required fields abort analysis
        if isinstance(features, dict):
            required = [
                'Air temperature [K]',
                'Process temperature [K]',
                'Rotational speed [rpm]',
                'Torque [Nm]',
                'Tool wear [min]',
                'Type',
            ]
            if not all(key in features for key in required):
                return []

            air_temp = float(features['Air temperature [K]'])
            process_temp = float(features['Process temperature [K]'])
            rpm = float(features['Rotational speed [rpm]'])
            torque = float(features['Torque [Nm]'])
            wear = float(features['Tool wear [min]'])
            machine_type = str(features['Type'])
        elif isinstance(features, pd.DataFrame):
            row = features.iloc[0]
            required_sensor_cols = [
                'Air temperature [K]',
                'Process temperature [K]',
                'Rotational speed [rpm]',
                'Torque [Nm]',
                'Tool wear [min]',
            ]
            if not all(col in features.columns for col in required_sensor_cols):
                return []

            air_temp = float(row['Air temperature [K]'])
            process_temp = float(row['Process temperature [K]'])
            rpm = float(row['Rotational speed [rpm]'])
            torque = float(row['Torque [Nm]'])
            wear = float(row['Tool wear [min]'])

            # Determine machine type from one-hot encoded columns or explicit Type column.
            if 'Type_L' in features.columns and float(row['Type_L']) == 1.0:
                machine_type = 'L'
            elif 'Type_M' in features.columns and float(row['Type_M']) == 1.0:
                machine_type = 'M'
            elif 'Type_H' in features.columns and float(row['Type_H']) == 1.0:
                machine_type = 'H'
            elif 'Type' in features.columns and str(row['Type']) in {'L', 'M', 'H'}:
                machine_type = str(row['Type'])
            else:
                return []
        else:
            # Array format - assume standard feature order
            return []
        
        failure_modes = []
        
        # TWF: Tool Wear Failure (wear > 200 min)
        if wear > 200:
            wear_excess = wear - 200.0
            risk = self._empirical_risk(
                wear_excess,
                self.risk_calibration.get("twf_wear_excess")
            )
            failure_modes.append({
                "mode": "TWF",
                "name": "Tool Wear Failure",
                "risk": risk,
                "trigger": f"Tool wear {wear:.1f} min exceeds 200 min threshold"
            })
        
        # HDF: Heat Dissipation Failure (temp_diff < 8.6K AND rpm < 1380)
        temp_diff = process_temp - air_temp
        if temp_diff < 8.6 and rpm < 1380:
            temp_margin = 8.6 - temp_diff
            rpm_margin = 1380.0 - rpm
            risk_temp = self._empirical_risk(
                temp_margin,
                self.risk_calibration.get("hdf_temp_margin")
            )
            risk_rpm = self._empirical_risk(
                rpm_margin,
                self.risk_calibration.get("hdf_rpm_margin")
            )
            risk = (risk_temp + risk_rpm) / 2.0
            failure_modes.append({
                "mode": "HDF",
                "name": "Heat Dissipation Failure",
                "risk": risk,
                "trigger": f"Temp diff {temp_diff:.1f}K < 8.6K AND RPM {rpm:.0f} < 1380"
            })
        
        # PWF: Power Failure (power < 3500W or > 9000W)
        power = torque * rpm * np.pi / 30.0  # Convert to Watts
        if power < 3500 or power > 9000:
            if power < 3500:
                power_excess = 3500.0 - power
                risk = self._empirical_risk(
                    power_excess,
                    self.risk_calibration.get("pwf_low_excess")
                )
            else:
                power_excess = power - 9000.0
                risk = self._empirical_risk(
                    power_excess,
                    self.risk_calibration.get("pwf_high_excess")
                )
            failure_modes.append({
                "mode": "PWF",
                "name": "Power Failure",
                "risk": risk,
                "trigger": f"Power {power:.0f}W outside 3500-9000W range"
            })
        
        # OSF: Overstrain Failure (wear × torque exceeds type limit)
        type_limits = {"L": 11000, "M": 12000, "H": 13000}
        if machine_type not in type_limits:
            return failure_modes

        type_limit = type_limits[machine_type]
        overstrain_value = wear * torque
        if overstrain_value > type_limit:
            overstrain_excess = overstrain_value - type_limit
            osf_excess_by_type = self.risk_calibration.get("osf_excess_by_type", {})
            risk = self._empirical_risk(
                overstrain_excess,
                osf_excess_by_type.get(machine_type)
            )
            failure_modes.append({
                "mode": "OSF",
                "name": "Overstrain Failure",
                "risk": risk,
                "trigger": f"Wear×Torque {overstrain_value:.0f} minNm exceeds type {machine_type} limit {type_limit}"
            })
        
        return failure_modes