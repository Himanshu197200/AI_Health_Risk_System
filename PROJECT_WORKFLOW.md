# Project Workflow — Detailed Architecture & Engineering Blueprint

This document serves as the exhaustive technical manual mapping the internal mechanics of the **Intelligent Patient Risk Assessment and Agentic Health Support System**. It is designed to trace data accurately from raw CSV inputs entirely through the Machine Learning inference boundary, and out into the Large Language Model (LLM) conversational interface. 

By detailing this unified infrastructure, the pipeline bridges the complex gap between automated mathematical probability scoring and safe, pragmatic human healthcare advisory.

---

## Expanded Systems Architecture

Our project is decoupled into two primary temporal stages: **The Engine / Training Pipeline** (which builds our computational models locally), and the **Inference Dashboard** (which executes real-time user-data against an Agentic AI).

### Phase 1: Machine Learning Training Flow

```text
       [ Raw Clinical CSV (97.2K rows) ]
                 │
                 ▼
 ┌───────────────────────────────┐     ┌──────────────────────────────────┐
 │    preprocess.py (Cleaner)    │ ──▶ │ features.py (Standardization)    │
 │ ----------------------------- │     │ -------------------------------- │
 │ 1. Clean missing outliers     │     │ 1. Initialize StandardScaler     │
 │ 2. One-Hot Encode categories  │     │ 2. Fit ONLY to 80% Training      │
 │ 3. Slice 80-20 Train/Test DB  │     │ 3. Transform Testing & Metrics   │
 └───────────────────────────────┘     └──────────────────────────────────┘
                 │                                       │
                 ▼                                       ▼
        ┌─────────────────┐                     ┌─────────────────┐
        │    train.py     │ ◀────────────────── │  evaluate.py    │
        │ --------------- │                     │ --------------- │
        │ Fits target to: │                     │ Validates via:  │
        │  * Linear Reg   │ ──(Score loops)──▶  │  * MAE & MSE    │
        │  * Dec. Tree    │                     │  * RMSE & R²    │
        └─────────────────┘                     └─────────────────┘
                 │                                       │
                 ▼                                       ▼
 ┌────────────────────────────────────────────────────────┐
 │                      models/                           │
 │ (Successfully saves winning Joblib binary `.pkl` files)│
 └────────────────────────────────────────────────────────┘
```

### Phase 2: Agentic Inference & Dashboard Flow

```text
 [ Streamlit User UI Slider Data ] ──▶ Evaluated against 5 saved `.pkl` Regressors
                 │
                 ▼
 [ 5x Clinical Risk Output Scores (Heart, Diabetes, Hypertension, Obesity, Cholesterol) ]
                 │
                 ▼
 ┌──────────────────────────────────────────────────────────┐
 │ health_agent.py (The Orchestrator Logic)                 │
 │ -------------------------------------------------------- │
 │ Checks 5 conditions -> Determines "High" & "Med" Threats │
 └──────────────────────────────────────────────────────────┘
                 │
                 ▼ (Calls Similarity Search on Medical Literature)
 ┌──────────────────────────────────────────────────────────┐     ┌─────────────────────┐
 │ knowledge_base.py (Retrieval-Augmented Generation / RAG) │ ◀── │ FAISS Vector Stores │
 │ -------------------------------------------------------- │     └─────────────────────┘
 │ 1. Ingests threat flags (e.g. "pre-diabetic")            │
 │ 2. Converts text to vectors via HuggingFace Embeddings   │
 │ 3. Returns WHO/CDC semantic literature chunks securely   │
 └──────────────────────────────────────────────────────────┘
                 │
                 ▼ (Compiles Huge Context Block prompting the Generative AI)
 ┌──────────────────────────────────────────────────────────┐
 │ llm_client.py (Generative Cloud Gateway)                 │
 │ -----------------------------------------                │
 │ Pings the 'Groq API' targeting 'Llama-3.1-70B' with the  │
 │ Patient Metrics + Strict System Prompt + RAG text bounds │
 └──────────────────────────────────────────────────────────┘
                 │
                 ▼
 [ Markdown AI Report / HTML Plotly Gauges / Automated FPDF File Generation ]
```

---

## Exhaustive Structural Pipeline Breakdown

### 1. Data Foundation (`data/`)

* **`data/health_risk_predictor.csv`:** The heart of our evaluation truth. Contains over ~97,000 distinct patient profile records spanning 28 comprehensive lifestyle/disease columns. Because this target scales linearly, we inherently extract continuous scoring metrics mapping target fields for our five parameters natively (Diabetes, Heart Disease, Hypertension, Obesity, Cholesterol).
* **`data/vector_db/`:** A persisted FAISS semantic directory. Instead of relying on traditional SQL search clauses, this database maps complex health documents algebraically—allowing our AI agent to semantically "understand" that 'high glucose levels' correlate contextually to vectors labeled 'Type 2 prevention techniques.' 

---

### 2. The Predictive Brain (`src/models/` & `src/`)

**`src/preprocess.py`**
* Before feeding an algorithm logic, text matrices (like `smoking_status: "Current" or "Former"`) must be converted to binary boolean flags (1/0) using **One-Hot Encoding**, preventing ML matrix calculations from crashing. 
* It isolates target parameters specifically away from predictors.

**`src/features.py`**
* Responsible for instantiating the foundational statistical boundaries mapping variables across massive dimensional bounds (like BMI bounds between ~15-40 versus Glucose Bounds spanning 60-350+). 
* To prevent explicit mathematical **Data Leakage** leading to false positives, the `StandardScaler` establishes the data "fit" solely against the isolated 80% independent training set, applying those calculated coefficients over the blind testing set globally.

**`src/train.py` & `src/evaluate.py`**
* Cycles iteratively across two statistical architectures independently for each of the five diseases. **Linear Regression** formulates global coefficients prioritizing interpretable feature magnitude offsets, whereas the **Decision Tree Regressors** structure complex boundary partitions against correlated health indicators (i.e. parsing blood pressure specifically relative to patient weight matrices).
* `evaluate.py` rigorously compares resulting algorithmic outputs tracking directly against **RMSE** (penalizing significant prediction errors drastically) resulting frequently in Linear Regression models outperforming trees generally around $R^2 \approx 0.95$, validating the hypothesis that metabolic systems natively scale linearly.

---

### 3. The Generative Architecture (`src/agent/`)

While predicting biological danger limits is powerful, passing raw calculated coefficients back to unprepared patients poses ethical and clinical failure points. The Agentic framework synthesizes numbers dynamically into textual, human-actionable empathy logs specifically via **RAG (Retrieval-Augmented Generation)**.

**`src/agent/knowledge_base.py` ("The Memory")**
* If absent natively, compiles raw textual information (from WHO guidelines) dynamically using `RecursiveCharacterTextSplitter`. 
* Passes fragmented chunks securely through HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`) locally, bypassing dependency over external paid models or external internet connections to vectorize language boundaries mapping to clinical guidance effectively. 

**`src/agent/llm_client.py` ("The Voice")**
* The external network orchestrator pinging the high-speed LPU infrastructure inside **Groq**. 
* Governs the central "System Prompt." This explicit hardcoded parameter string drastically bounds the `Llama 3.1 70B` model from attempting outright diagnoses, legally compelling the agent to generate disclaimers whilst actively mapping its generation securely inside empathetic "advice-modes" explicitly.

**`src/agent/health_agent.py` ("The Controller logic")**
* Initiates the overarching agent pipeline. When `app.py` passes the 5 disease parameters alongside raw patient vitals natively over to this class, the orchestrator script filters purely the parameters scoring inside "Medium" or "High" threat bands.
* Uses those threatening bands to execute a local similarity retrieval against the FAISS Memory array securely, compiling those localized RAG results underneath the patient UI variables into a massive, heavily verified text prompt payload for the Llama iteration effectively curtailing any "AI hallucinations".

---

### 4. Interactive Utilities & Rendering (`src/utils/` & `app.py`)

**`app.py` (The Interactive Dashboard)**
* Designed explicitly using `Streamlit` frameworks, parsing natively over complex tabbed interface groupings.
* Integrates python decorater caching (`@st.cache_resource`) bounding ML model inference endpoints directly. This essentially suspends large joblib `.pkl` weights in memory natively, avoiding extensive CPU delays during successive UI slider toggles globally.

**`src/utils/visualization.py`**
* Synthesizes isolated graphical widgets mapping exclusively against `Plotly`. Outputs reactive circular gauges scaling boundaries against (Green / Low, Yellow / Medium, Red / High) mapping 0-100 logic natively.

**`src/utils/pdf_generator.py`**
* Provides localized session persistence by overriding standard `FPDF` structures. Parses generated LLM conversational markdown natively out into printable artifacts bounding formatting breaks natively preserving records for external clinical doctor engagements.

---

## Central Structural Philosophies

1. **Strictly Decoupled Reasoning:** Machine Learning solely calculates probabilities. The LLM solely generates contextual strings. The integration strictly passes one into the other, preventing overlap or recursive failures.
2. **Defensive AI (RAG Constraints):** Never trust an LLM parameter model blindly inside a medical domain. All conversational advice generated within the application corresponds purely and exclusively through external local indexing lookups.
3. **Session Encapsulation:** Streamlit completely isolates independent user requests mitigating session bleeding natively through internal architecture configurations mimicking standard server-side application endpoints.

---

## End-To-End Execution Workflow

1. **Patient Arrives:** Adjusts the sidebar inputs (e.g., changes Age to 45, Systolic BP to 140, etc.).
2. **Analysis Triggered:** Clicks "Analyze Risks".
3. **Internal Prediction Phase:** System normalizes these inputs via the saved scaler and executes `model.predict()` loops dynamically over the 5 loaded disease `.pkl` files in memory entirely within milliseconds.
4. **Data Plotting:** Updates Dashboard, plotting Gauges visualizing raw statistical models.
5. **Generative Loop:** Agent checks the risk. Sees Hypertension is >66 (High Risk).
6. **RAG Vector Search:** Agent searches FAISS memory for "Hypertension Management", retrieving DASH diet protocols and blood-pressure strategies.
7. **Synthesis:** Appends context, prompts the Llama-70B model via Groq, generating comprehensive text dynamically mapping over to the "AI Health Report" interface. 
8. **Exportation:** User hits "Download PDF"—compiling all text logs into formatted export sheets seamlessly.
