# Anywhere Agentic AI - Predictive Maintenance System

<img width="1905" height="902" alt="Screenshot 2026-05-08 000256" src="https://github.com/user-attachments/assets/c6a8bcb2-b9b8-48bc-af48-0c96bb721f2a" />


<img width="1919" height="927" alt="Screenshot 2026-05-08 000525" src="https://github.com/user-attachments/assets/9689d2ab-f2c2-4295-a28f-5f58deb74ee1" />

## 📋 Overview

**Anywhere** is an intelligent predictive maintenance system powered by LangChain-based AI agents. It enables users to interact with machine failure data through a conversational interface, leveraging machine learning models and reasoning-based agents to predict equipment failures and recommend maintenance actions.

### Key Features
- **Conversational AI Agent**: ReAct-based agent for reasoning about machine failures
- **Multi-Model ML Pipeline**: Random Forest, Gradient Boosting, Logistic Regression, and more
- **Real-time Predictions**: Predict equipment failure risk with confidence scores
- **Failure Mode Analysis**: Identifies specific failure modes (Tool Wear, Heat Dissipation, Power, Overstrain, Random)
- **Interactive Dashboard**: Streamlit-based web interface with industrial dark theme
- **Multiple LLM Providers**: Support for Groq, Google Gemini, and DeepSeek
- **Rich Visualizations**: Interactive charts and detailed analysis results

---

## 🏗️ Project Structure

```
AnywhereAgenticAI/
├── app.py                      # Main entry point (Streamlit app)
├── ml_init.py                  # ML initialization
├── pages.py                    # Chat interface page component
├── sidebar.py                  # Agent configuration sidebar
├── styles.py                   # Custom CSS styling
├── MaiStorage.csv              # Milling machine dataset
├── requirements.txt            # Python dependencies
│
├── agent/
│   ├── __init__.py
│   └── maintenance_agent.py    # LangChain ReAct agent for maintenance reasoning
│
├── data/
│   ├── __init__.py
│   └── loader.py               # Data loading and processing utilities
│
└── models/
    ├── __init__.py
    └── predictor.py            # ML model training and prediction
```

---

## 📄 File Descriptions

### **Core Application**

#### `app.py`
**Purpose**: Main entry point for the Streamlit web application  
**Responsibilities**:
- Configures the Streamlit page layout and settings
- Injects custom CSS styling
- Renders the sidebar for agent configuration
- Renders the main chat interface
- Coordinates the UI components and initializes the application
**Key Functions**: `main()`

#### `pages.py`
**Purpose**: Renders the agent chat interface and conversation management  
**Responsibilities**:
- Displays the chat message history
- Handles user input and agent responses
- Renders visualizations from tool predictions
- Formats and displays tables, charts, and analysis results
- Manages conversation flow between user and AI agent
**Key Functions**:
- `render_agent_chat_page()`: Main chat interface renderer
- `dataframe_to_html_table()`: Converts DataFrames to styled HTML tables
- `_render_visualization()`: Renders tool-specific visualizations

#### `sidebar.py`
**Purpose**: Renders the agent configuration panel in the sidebar  
**Responsibilities**:
- Displays system status indicators
- Allows users to select LLM provider (Groq, Gemini, DeepSeek)
- Shows model loading status and agent information
- Displays dataset statistics and model accuracy
**Key Functions**: `render_agent_sidebar()`

#### `styles.py`
**Purpose**: Custom CSS styling for the industrial dark theme  
**Responsibilities**:
- Injects custom CSS into the Streamlit app
- Defines color scheme (electric cyan, amber, red, green accents)
- Styles components for consistent visual design
- Provides responsive design rules
- Sets font families (Orbitron, Share Tech Mono, Exo 2)

---

### **AI Agent**

#### `agent/maintenance_agent.py`
**Purpose**: LangChain-based ReAct agent for maintenance reasoning and failure analysis  
**Responsibilities**:
- Initializes the AI agent with failure detection thresholds
- Integrates multiple tools for machine analysis (predict_failure, analyze_wear, etc.)
- Manages conversation memory and multi-step reasoning
- Handles different LLM providers (Groq, Google Gemini, DeepSeek)
- Trains and manages predictive models
- Returns visualizations alongside predictions

**Key Features**:
- **Failure Mode Thresholds**:
  - Tool Wear Failure (TWF): 200-240 min wear
  - Heat Dissipation Failure (HDF): Temp diff < 8.6K, RPM < 1380
  - Power Failure (PWF): 3500-9000 Watts
  - Overstrain Failure (OSF): Type-dependent limits (11000-13000 minNm)
  - Random Failure (RNF): 0.1% probability

**Key Functions**:
- `__init__()`: Initialize agent with dataset and LLM provider
- `_initialize_llm()`: Set up LLM provider connection
- `run()`: Execute agent with user query
- `predict_failure()`: Tool to predict equipment failure
- `analyze_maintenance()`: Tool to analyze maintenance needs

---

### **Data Processing**

#### `data/loader.py`
**Purpose**: Load and process the milling machine dataset  
**Responsibilities**:
- Load CSV data from `MaiStorage.csv`
- Provide dataset summary statistics
- Calculate failure rates and metrics
- Integrate with ML model for quick accuracy checks

**Key Functions**:
- `load_dataset()`: Load the milling machine CSV
- `get_summary_stats()`: Return dataset statistics (total samples, failures, failure rate, model accuracy)

---

### **Machine Learning**

#### `models/predictor.py`
**Purpose**: ML model training and failure prediction  
**Responsibilities**:
- Train multiple classifier models on historical failure data
- Select the best performing model automatically
- Make predictions on new equipment states
- Calculate confidence scores and probability calibration
- Handle feature engineering and scaling

**Supported Models**:
- Random Forest Classifier
- Gradient Boosting Classifier
- Logistic Regression
- Decision Tree Classifier
- Neural Network MLP

**Features Used** (to avoid data leakage):
- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]
- Machine type (L, M, H)

**Key Functions**:
- `train_all()`: Train multiple models and select best one
- `predict()`: Make predictions with confidence scores
- `get_feature_importance()`: Return feature importance metrics

#### `ml_init.py`
**Purpose**: ML module initialization  
**Status**: Currently minimal - imports fraud detector

---

### **Dataset**

#### `MaiStorage.csv`
**Purpose**: Milling machine operational data with failure labels  
**Contains**:
- Machine features (air temp, process temp, RPM, torque, tool wear)
- Machine type (L, M, H)
- Failure mode indicators (TWF, HDF, PWF, OSF, RNF)
- Binary failure target (0 = normal, 1 = failure)

---

## 🔄 Application Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER OPENS BROWSER                           │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    app.py (Main Entry)                          │
│ - Initialize Streamlit page config                             │
│ - Inject custom CSS from styles.py                             │
│ - Render sidebar & main content                                │
└─────────────┬───────────────────────────────────┬───────────────┘
              │                                   │
              ▼                                   ▼
    ┌──────────────────┐            ┌──────────────────────────┐
    │  sidebar.py      │            │     pages.py             │
    │ ──────────────   │            │ ──────────────────────   │
    │ - LLM Provider   │            │ - Chat History Display  │
    │ - System Status  │            │ - User Input Box        │
    │ - Model Info     │            │ - Agent Response        │
    │ - Dataset Stats  │            │ - Visualizations        │
    └────────┬─────────┘            └──────────┬───────────────┘
             │                                 │
             │ (User selects LLM)              │ (User types query)
             │                                 │
             └────────────┬────────────────────┘
                          │
                          ▼
      ┌────────────────────────────────────────────────┐
      │  maintenance_agent.py (ReAct Agent)            │
      │ ──────────────────────────────────────────    │
      │ - Parse user query                            │
      │ - Select appropriate tools                    │
      │ - Execute reasoning loop                      │
      │ - Gather results & visualizations             │
      └────┬──────────────────────────┬───────────────┘
           │                          │
           ▼                          ▼
    ┌─────────────────┐        ┌─────────────────────┐
    │ data/loader.py  │        │ models/predictor.py │
    │ ─────────────   │        │ ─────────────────   │
    │ - Load CSV data │        │ - Train models      │
    │ - Calculate     │        │ - Make predictions  │
    │   statistics    │        │ - Calculate scores  │
    │ - Feature prep  │        │ - Get importance    │
    └─────────────────┘        └─────────────────────┘
           │                          │
           └────────────┬─────────────┘
                        │
                        ▼
      ┌──────────────────────────────────────┐
      │  Results with Visualizations         │
      │ - Failure Risk Score                 │
      │ - Prediction (NORMAL/FAILURE)        │
      │ - Confidence Level                   │
      │ - Feature Analysis Charts            │
      │ - Maintenance Recommendations        │
      └──────────────────────────────────────┘
                        │
                        ▼
              ┌──────────────────────┐
              │   Display in Chat    │
              │   (pages.py)         │
              └──────────────────────┘
```

---

## 🔧 Technology Stack

### Core Framework
- **Streamlit**: Web application framework
- **LangChain**: AI agent framework with ReAct prompting
- **Pandas & NumPy**: Data processing and numerical computation

### Machine Learning
- **Scikit-learn**: ML model implementations and metrics
- **Plotly**: Interactive visualizations

### LLM Providers
- **Groq**: Fast inference with llama models
- **Google Generative AI**: Gemini models
- **OpenRouter/DeepSeek**: Additional LLM options

### Utilities
- **python-dotenv**: Environment variable management
- **Pydantic**: Data validation

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation

1. **Clone and navigate to project**:
   ```bash
   cd AnywhereAgenticAI
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1  # Windows PowerShell
   # or
   source .venv/bin/activate      # Linux/Mac
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** (create `.env` file):
   ```
   GROQ_API_KEY=your_groq_api_key
   GOOGLE_API_KEY=your_google_api_key
   OPENROUTER_API_KEY=your_openrouter_api_key
   ```

5. **Run the application**:
   ```bash
   streamlit run app.py
   ```

6. **Access in browser**:
   ```
   http://localhost:8501
   ```

---

## 🔑 Key Concepts

### Failure Modes
The system detects five types of equipment failures:

1. **Tool Wear Failure (TWF)**: Excessive tool wear (200-240 min)
2. **Heat Dissipation Failure (HDF)**: Temperature difference too high or RPM too low
3. **Power Failure (PWF)**: Power consumption outside safe range (3500-9000W)
4. **Overstrain Failure (OSF)**: Wear × torque product exceeds type limits
5. **Random Failure (RNF)**: Stochastic failure event (0.1% probability)

### Prediction Pipeline
1. **Feature Extraction**: Load sensor data from CSV
2. **Model Training**: Train multiple models, select best performer
3. **Risk Calibration**: Empirical probability calibration from historical data
4. **Prediction**: Generate failure risk score and confidence level
5. **Visualization**: Present results with interactive charts and recommendations

### Agent Reasoning
The ReAct agent follows a reasoning loop:
1. **Observe**: Receive user query
2. **Think**: Determine which tools to use
3. **Act**: Execute selected tools
4. **Reflect**: Analyze results and decide next steps
5. **Response**: Provide final answer to user

---

## 📊 Example Queries

- "What's the failure risk for equipment with 230 min tool wear?"
- "Analyze the current machine state and recommend maintenance"
- "Compare failure patterns across different machine types"
- "What factors contribute most to equipment failures?"
- "Is this equipment approaching a maintenance threshold?"

---

## 🔐 Security Notes

- Never commit `.env` file with API keys
- Redact sensitive information in logs
- Store credentials only in environment variables or secure vaults

---

## 📝 License & Attribution

Developed for predictive maintenance use case. Based on milling machine operational data.

---

## 🤝 Support

For issues or feature requests, check the project repository or contact the development team.

---

**Last Updated**: May 2026  
**Version**: 1.0
