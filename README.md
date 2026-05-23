Python
markdown_content = """# 🎮 FPS Oracle: Production-Ready Gaming Frame Rate Predictor

![Python Version](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Neon_Cloud-336791?style=for-the-badge&logo=postgresql&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-F37626?style=for-the-badge&logo=xgboost&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.7.2-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

An end-to-end Machine Learning web ecosystem engineered to predict in-game frame rates (FPS) by evaluating complex hardware synergies (CPU/GPU) against real-time software configurations. 

This repository represents a fully operational **Proof of Concept (PoC)** demonstrating a modular, decoupled software engineering architecture: featuring a high-performance machine learning pipeline, cloud-hosted relational database layer, custom-compiled user interface, and cross-environment deployment paradigms.

---

## 🏗️ System Architecture & Data Flow

The application is architected using a decoupled design pattern where the compute layer, data storage layer, and client presentation layer communicate seamlessly via stateless protocols and managed environment injections.

Code output
Code executed successfully!
```mermaid
graph TD
    A[Streamlit Client UI] -->|1. Request Dropdown Catalog| B(SQLAlchemy Engine)
    B -->|2. Execute Textual SQL Query| C[(PostgreSQL Cloud Instance - Neon)]
    C -->|3. Return Unique Hardware Records| B
    B -->|4. Populate Options| A
    A -->|5. Trigger Predict Inference| E[Data Assembly & Preprocessing]
    E -->|6. Enforce Feature Column Alignment| F{Serialized Sklearn Pipeline .pkl}
    F -->|7. Multi-Channel Column Transformation| G[XGBoost Estimator Engine]
    G -->|8. Generate Clamped Float Scalar| A
Technical Design Specs:
The Client Layer (Streamlit): Serves as a responsive web dashboard injected with custom CSS tokens (Orbitron, Rajdhani typography) and dark cyberpunk themes. It implements high-efficiency caching decorators (@st.cache_resource, @st.cache_data) with structured Time-To-Live (ttl=600) parameters to eliminate redundant database handshakes.

The Database Catalog Layer (PostgreSQL): Hosted on an isolated serverless cloud instance (Neon.tech). Contains separate relations for cpus, gpus, games, and game_settings.

The Inference Processing Engine (Scikit-Learn/XGBoost): A serialized computing graph loaded lazily via joblib, executing complex vector transformations in memory.

🧠 Machine Learning Pipeline Deep Dive
The analytical engine utilizes a robust composite architecture consisting of multi-stage column transformers and non-linear regression boosting trees.

Input Vector (Raw Features)
       │
       ├───► [Numerical Block] ───► SimpleImputer(median) ───► StandardScaler() ───┐
       │                                                                           ▼
       ├───► [Low-Card Cat] ──────► SimpleImputer(frequent) ─► OneHotEncoder() ───┼─► [Concatenated Matrix] ─► XGBRegressor
       │                                                                           ▲
       └───► [High-Card Cat] ─────► SimpleImputer(frequent) ─► TargetEncoder() ───┘
1. Multi-Channel Preprocessing Framework
To process raw system telemetry without human intervention, the features are partitioned into isolated computational tracks through an explicit ColumnTransformer:

Numerical Normalization: Strategic hardware parameters (CpuCores, CpuThreads, CpuBaseClock, CpuBoostClock, CpuTDP, GpuVRAM, GpuBandwidth, GpuTDP, GpuBoostClock) undergo a median-imputation pass via SimpleImputer(strategy='median') to handle missing spec data gracefully, followed by a StandardScaler() alignment to preserve gradient descent efficiency.

Low-Cardinality Categorical Encoding: Features with minimal feature spaces (e.g., GameSetting variants like Low, Medium, High, Ultra) are explicitly mapped into a binary array layout using a sparse-disabled OneHotEncoder(handle_unknown='ignore').

High-Cardinality Categorical Encoding: Structural fields with massive, unstructured text spaces (CpuName, GpuName, GameName) are dynamically encoded using an advanced supervised TargetEncoder(). This calculates the structural weights of each hardware model relative to the continuous target output value (FPS), preventing the dimensionality explosion typical of standard one-hot encoding.

2. Gradient Boosted Estimator Core
The system utilizes an optimized XGBRegressor equipped with tuned hyperparameters (n_estimators=200, learning_rate=0.05, max_depth=7) designed to model severe non-linear bottlenecks (e.g., CPU bottlenecking high-end GPUs or thermal throttle simulations).

3. Production Environment Compatibility Fixes
Version Control Alignment: To mitigate deserialization breaks during unpickling (AttributeError: 'SimpleImputer' object has no attribute '_fill_dtype'), the runtime environment explicitly tracks and standardizes core computational libraries on scikit-learn==1.7.2.

Byte-String Cleansing Layer: A dedicated regex/string filtering utility (clean_name) is implemented directly within the UI layout to strip trailing legacy byte-literal encodings (b'...') natively parsed from historical Kaggle benchmark sheets.

🗄️ Relational Database & Migration Blueprint
The system uses SQLAlchemy 2.0 to manage relational abstractions. To bypass strict case-insensitivity default bindings inside PostgreSQL which automatically fold identifiers to lowercase, the queries implement explicit double-quoted schema mapping ("CpuName", "GpuName").

Schema Structure
SQL
CREATE TABLE cpus (
    "CpuName" TEXT PRIMARY KEY,
    "CpuCores" INTEGER,
    "CpuThreads" INTEGER,
    "CpuBaseClock" REAL,
    "CpuBoostClock" REAL,
    "CpuTDP" REAL
);

CREATE TABLE gpus (
    "GpuName" TEXT PRIMARY KEY,
    "GpuVRAM" REAL,
    "GpuBandwidth" REAL,
    "GpuTDP" REAL,
    "GpuBoostClock" REAL
);
🚀 Execution & Deployment Manual
1. Local Environment Provisioning
Initialize an isolated runtime wrapper and deploy the explicit backward-compatible computing stack:

Bash
# Initialize Python virtual environment
python -m venv venv
source venv/bin/activate  # Windows execution: venv\\Scripts\\activate

# Enforce explicit version-controlled dependency mapping
pip install scikit-learn==1.7.2 xgboost lightgbm pandas streamlit sqlalchemy psycopg2-binary python-dotenv category_encoders joblib
2. Database Migration Injection
Populate your remote serverless Postgres layer directly from local comma-separated values (.csv) sheets using the migration script:

Bash
python seed_db.py
3. Runtime Environment Variables
Construct a secure .env instance in the root execution folder:

Code snippet
DB_URL="postgresql+psycopg2://neondb_owner:npg_XhGnCsI2E4De@ep-ancient-snow-apkuy3od-pooler.c-7.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
MODEL_PATH="fps_predictor_pipeline.pkl"
4. Running the Stack
Launch the Streamlit orchestration server locally:

Bash
streamlit run App.py
5. Streamlit Cloud Secret Management
When pushing your repository to production on Streamlit Cloud, inject your variables securely via the App Secrets Dashboard using the following valid TOML configuration:

Ini, TOML
DB_URL = "postgresql+psycopg2://neondb_owner:npg_XhGnCsI2E4De@ep-ancient-snow-apkuy3od-pooler.c-7.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
📊 Dataset Note & Disclaimer
⚠️ Dataset Phase Notice: The current analytical model utilizes a legacy hardware benchmarking dataset as a robust Proof of Concept (PoC) to validate the end-to-end continuous data integration pipeline, relational table caching, and automated cloud queries. The operational foundation is immutable; subsequent phases will retrain the underlying XGBoost model on modern current-gen hardware (RTX 50-series / Ryzen 9000 series) without requiring architectural alterations.

👨‍💻 Engineering Identity
Lead Engineer: Ahmed Hamdy

Professional Profile: Data Scientist & Machine Learning Engineer

Source Management: @a7med-830

📝 Appendix: Project Presentation Scripts
Option A: Technical Video Pitch (English)
Delivery Cadence: Speak confidently. Keep the Streamlit UI visible on screen and execute a demo run precisely as you state the word "instantly predicts".

"Guessing a PC's gaming performance shouldn't be a gamble. That’s exactly why I built FPS Oracle. It’s an end-to-end Machine Learning web app where you simply select a CPU, GPU, game title, and graphics preset, and the AI instantly predicts your exact frame rate and grades the system's performance.

Under the hood, this is a complete, production-ready pipeline divided into three main parts:

For the Machine Learning: I built the predictive model using Scikit-Learn and XGBoost. It includes complex data preprocessing—like custom imputation and target encoding—so the model truly understands hardware synergies.

For the Backend: There are no local files. The app dynamically fetches hardware specs from a live PostgreSQL database hosted on the cloud via Neon.

For the Frontend: I built the interface using Streamlit, injecting custom CSS to give it a dark, cyberpunk aesthetic that fits the gaming vibe.

Now, full transparency: the dataset currently powering the model relies on slightly older hardware benchmarks. I used this specific data as a Proof of Concept to validate the end-to-end architecture, the ML pipeline, and the cloud database integration. The foundation is solidly built, so the next step is simply retraining the model with the latest benchmark data for current-gen hardware.

This project was a great challenge bridging Data Science, Backend engineering, and Cloud Deployment. I’ll leave the GitHub link below if you want to check out the code. I'd love to hear your thoughts or feedback!"
