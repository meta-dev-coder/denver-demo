from google.api_core import exceptions
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import google.generativeai as genai  # Swapped from Groq to Gemini

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Denver Climate Digital Twin v2", layout="wide")

# Initialize LLM Client (Replace with your API Key or use Streamlit Secrets)
# Get your free key at console.groq.com
# client = Groq(api_key=st.secrets.get("GROQ_API_KEY", "None"))
api_key = st.secrets.get("GEMINI_API_KEY", "None")
genai.configure(api_key=api_key)

# List of models to try in order of preference
MODEL_LIST = [
    'gemini-2.5-flash-lite',       # Primary
    'gemini-3-flash-preview',# Reliable backup
    'gemini-3-pro-preview'     # High-quota backup
]

# model = genai.GenerativeModel('gemini-2.0-flash')
# client = OpenAI(
#     api_key='None',
#     base_url="https://api.x.ai/v1",
# )
# --- DATA INGESTION (ACTUAL DENVER DATASETS) ---

# --- CACHED LOGIC ---
# This saves results so if you click the button twice, it doesn't use a request.
@st.cache_data(ttl=3600)  # Results stay valid for 1 hour
def get_llm_response(prompt, model_name):
    """Calls the API with a mandatory cooldown and error handling."""
    # 1. Mandatory Throttling: Stay under 15 RPM (1 request / 4 seconds)
    import time
    time.sleep(4)

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except exceptions.ResourceExhausted:
        # Pass the error up so we can trigger the fallback
        raise exceptions.ResourceExhausted("Rate limit reached.")
    except Exception as e:
        return f"Error: {str(e)}"



@st.cache_data
def load_denver_data():
    # Dataset 1: Tree Canopy by Neighborhood (Actual ArcGIS Link)
    canopy_url = "ODC_ENV_TREECANOPY2020_A_-5786833496851207437.csv"
    # Dataset 2: Neighborhood Demographic/Heat Vulnerability Proxy
    # vuln_url = "https://opendata.arcgis.com/datasets/4f305c48b7f14b6f9509a263158c973a_0.csv"

    canopy_df = pd.read_csv(canopy_url)
    return canopy_df
    # vuln_df = pd.read_csv(vuln_url)
    # return pd.merge(canopy_df, vuln_df, on="NBHD_NAME", how="inner")





df = None
try:
    df = load_denver_data()
except:
    st.error("Live data fetch failed. Check your internet or URLs.")
    st.stop()

# --- SIDEBAR & USER PROFILE ---
with st.sidebar:
    st.image("https://www.denvergov.org/content/dam/denvergov/Portal/logos/denver-logo.png", width=150)
    st.title("SuperDNA3D Lab")
    st.subheader("Denver Digital Twin Demo")
    st.info("User: Policy Analyst\nRole: Decision Support")
    app_mode = st.selectbox("Scenario Mode",
                            ["Overview", "Heat/Canopy Simulation",  "LLM Policy Advisor", "Electrification ML",])

# --- MODE 1: HEAT/CANOPY SIMULATION (ML) ---
if app_mode == "Heat/Canopy Simulation":
    st.header("🌳 Urban Heat Island (UHI) ML Surrogate")
    st.write("This surrogate model uses Denver Tree Canopy data to predict temperature reduction.")

    # Train a quick ML Model on the actual dataset
    # Features: Tree Canopy %, Pavement %, Target: Surface Temp (Proxy)
    X = df[['PCT_CANOPY']].fillna(0)
    y = 40 - (0.3 * X['PCT_CANOPY']) + np.random.normal(0, 0.5, len(X)) # Synthetic target for demo
    model = RandomForestRegressor(n_estimators=100).fit(X, y)

    col1, col2 = st.columns([1, 2])
    with col1:
        nbhd = st.selectbox("Select Neighborhood", df['NBHD_NAME'].unique())
        current_val = df[df['NBHD_NAME'] == nbhd]['PCT_CANOPY'].values[0]

        sim_canopy = st.slider("Target Canopy % Expansion", 0.0, 50.0, float(current_val))
        pred_temp = model.predict([[sim_canopy]])[0]

        st.metric("Predicted Peak Temp", f"{pred_temp:.1f}°C", delta=f"{pred_temp - 38:.1f}°C", delta_color="inverse")

    with col2:
        fig = px.scatter(df, x="PCT_CANOPY", y=y, hover_name="NBHD_NAME",
                         title="Live Denver Data: Canopy vs Predicted Surface Temp")
        st.plotly_chart(fig)

# --- MODE 2: LLM POLICY ADVISOR (ACTUAL LLM) ---
elif app_mode == "LLM Policy Advisor":
    st.header("🤖 Gemini Policy Analysis")
    st.write("Demonstrating RFP Task 4: Narrative Framing and Scenario Briefs")

    # Context derived from Denver's actual 80x50 Climate Action Plan
    context = """
    Denver's Climate Action Plan Goals:
    - 80% GHG reduction by 2050.
    - 20% tree canopy coverage in all residential neighborhoods by 2030.
    - Prioritize investments in Equity Priority Areas (Globeville, Elyria-Swansea).
    """

    user_query = st.text_area("Ask Gemini about policy alignment:",
                              "How does a 15% canopy increase in Globeville support Denver's 2030 goals?")

    if "current_model_idx" not in st.session_state:
        st.session_state.current_model_idx = 0

    if st.button("Generate Policy Brief"):
        success = False

        # Try models one by one starting from the current session's "working" model
        for i in range(st.session_state.current_model_idx, len(MODEL_LIST)):
            selected_model = MODEL_LIST[i]

            with st.spinner(f"Analyzing with {selected_model}..."):
                try:
                    result = get_llm_response(user_query, selected_model)
                    st.success(f"Generated using: {selected_model}")
                    st.markdown(result)
                    success = True
                    break # Exit loop on success

                except exceptions.ResourceExhausted:
                    st.warning(f"Quota exhausted for {selected_model}. Switching to backup...")
                    st.session_state.current_model_idx = i + 1 # Permanently switch for this session
                    continue # Try the next model in the list

        if not success:
            st.error("All available model quotas are currently exhausted. Please try again in 1 minute.")

# --- OVERVIEW ---
else:
    st.title("🏙️ Denver Digital Twin: Task 2 & 3 Demo")
    st.markdown("""
    ### Technical Alignment:
    - **Task 2 (Data):** Ingesting live ArcGIS CSVs from Denver Open Data.
    - **Task 3 (ML):** Running a Tier 1 Random Forest Surrogate for UHI prediction.
    - **Task 4 (Reporting):** Using Llama 3 for automated "Scenario Briefs".
    """)
    if df is not None:
        st.dataframe(df.head(10))