from google.api_core import exceptions
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import google.generativeai as genai  # Swapped from Groq to Gemini

from generate_5k_ds import DENVER_NBHDS

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
    flood_df  = pd.read_csv("denver_flood_5k.csv")
    return canopy_df, flood_df
    # vuln_df = pd.read_csv(vuln_url)
    # return pd.merge(canopy_df, vuln_df, on="NBHD_NAME", how="inner")

def train_surrogate(df):
    X = df[['Year', 'Month', 'Rainfall_In', 'SVI', 'Canopy_Pct']]
    y_depth = df['Depth_m']
    y_loss = df['Property_Loss_USD']

    model_depth = RandomForestRegressor(n_estimators=50).fit(X, y_depth)
    model_loss = RandomForestRegressor(n_estimators=50).fit(X, y_loss)
    return model_depth, model_loss

# --- NEW: AGENTIC INTENT PARSER ---
def parse_scenario_intent(user_input):
    """Uses LLM to turn text into slider values"""
    prompt = f"""
    Convert this user request into a JSON scenario object for a flood simulator.
    User Request: "{user_input}"
    Rules:
    - Year: between 2006 and 2026.
    - Month: 1 to 12. or in Jan-Dec or January-December format.
    - Rainfall_In: 0.0 to 8.0. can be in inches or "inch" format.
    - Target: "Depth" or "Loss".
    Return ONLY JSON: {{"year": 2026, "month": 6, "rain": 4.5, "target": "Loss"}}
    """
    if "current_model_idx" not in st.session_state:
        st.session_state.current_model_idx = 0
    import json
    for i in range(st.session_state.current_model_idx, len(MODEL_LIST)):
        selected_model = MODEL_LIST[i]

        with st.spinner(f"Analyzing with {selected_model}..."):
            try:
                result = get_llm_response(user_query, selected_model)
                # st.success(f"Generated using: {selected_model}")
                # st.markdown(result)
                success = True
                break # Exit loop on success

            except exceptions.ResourceExhausted:
                st.warning(f"Quota exhausted for {selected_model}. Switching to backup...")
                st.session_state.current_model_idx = i + 1 # Permanently switch for this session
                continue # Try the next model in the list

    if not success:
        st.error("All available model quotas are currently exhausted. Please try again in 1 minute.")
    # model = genai.GenerativeModel(MODEL_NAME)
    # response = model.generate_content(prompt)
    try:
        return json.loads(result.strip().replace('```json', '').replace('```', ''))
    except:
        return None

df = None
df_flood = None
try:
    df, df_flood = load_denver_data()

except:
    st.error("Live data fetch failed. Check your internet or URLs.")
    st.stop()

model_depth, model_loss = train_surrogate(df_flood)

# --- SIDEBAR & USER PROFILE ---
with st.sidebar:
    st.image("https://www.denvergov.org/content/dam/denvergov/Portal/logos/denver-logo.png", width=150)
    st.title("SuperDNA3D Lab")
    st.subheader("Denver Digital Twin Demo")
    st.info("User: Policy Analyst\nRole: Decision Support")
    app_mode = st.selectbox("Scenario Mode",
                            [
                             "Storm Network Simulation",
                             "Flood Resilience Simulation",
                             "Heat/Canopy Simulation",
                             "LLM Policy Advisor",
                             "Electrification ML",
                                "Overview"])

if "current_model_idx" not in st.session_state:
    st.session_state.current_model_idx = 0

# --- NEW MODE: STORM NETWORK SIMULATION ---
if app_mode == "Storm Network Simulation":
    st.header("🌊 Storm Sewer Hydraulic Twin")
    st.write("Mapping street-level rain to underground pipe capacity.")

    # 1. Agentic Prompting
    user_query = st.text_area("Describe a storm scenario:",
                              placeholder="Simulate a 4-inch flash flood in North Denver...")

    # 2. Controls
    c1, c2 = st.columns(2)
    runoff_c = c1.slider("Surface Runoff Coefficient (C)", 0.1, 0.95, 0.3,
                         help="0.9 = Concrete/Urban, 0.3 = Parks/Green Space")
    rain_i = c2.slider("Rainfall Intensity (Inches/Hr)", 0.0, 8.0, 1.0)

    # 3. Execution

    # --- METHODOLOGY SECTION ---
    with st.expander("📘 How it Works: Methodology & Mathematics"):
        st.markdown(r"""
        ### 1. The Rational Method (Surface Runoff)
        We calculate the peak flow ($Q$) entering each inlet based on the surface characteristics of its parent drainage basin:
        $$Q = C \cdot i \cdot A$$
        - **C**: Runoff Coefficient (User-defined via slider)
        - **i**: Rainfall Intensity (Inches/Hour)
        - **A**: Basin Area (Acres)

        ### 2. Manning’s Equation (Pipe Capacity)
        The capacity of the underground network is determined by pipe geometry and friction:
        $$V = \frac{1.486}{n} R_h^{2/3} S^{1/2} \quad \text{and} \quad Q_{cap} = A \cdot V$$
        - **n**: Roughness coefficient (0.013 for Concrete)
        - **Rh**: Hydraulic Radius ($D/4$ for full flow)
        - **S**: Pipe Slope ($ft/ft$)
        """)

    if st.button("Run Network Simulation"):
        import data_engine as de
        import llm_engine as le
        import sim_engine as se
        # Load datasets (mains, inlets, mapping)
        mains, mapping, basins = de.load_storm_network_data()

        # Build and Run
        G = se.build_storm_graph(mains)
        results_df = se.run_network_simulation(G, rain_i, runoff_c, mapping, basins)


        # UI Strategy for 34k records: Visualization over Tables
        st.subheader("System Performance")

        # Get Summary Data
        total_p, total_s, top_10 = se.get_simulation_summary(results_df)

        # --- DISPLAY TOTALS ---
        st.subheader("📍 Global Network Impact")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Pipes Analyzed", f"{total_p:,}")
        m2.metric("Total Surcharged Pipes", f"{total_s:,}", delta=f"{(total_s/total_p)*100:.1f}%", delta_color="inverse")
        m3.metric("System Health", f"{((total_p - total_s) / total_p)*100:.1f}%")

        # --- DISPLAY TOP 10 ---
        st.write("---")
        st.subheader("🔝 Top 10 Most Impacted Basins")

        col_chart, col_table = st.columns([2, 1])

        with col_chart:
            fig = px.bar(top_10, x='Basin', y='Surcharged_Count',
                         title="Critical Basins (Surcharge Count)",
                         color='Surcharged_Count', color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)

        with col_table:
            st.dataframe(top_10, hide_index=True, use_container_width=True)

        # AI Analysis: Only send the summary to avoid token limits
        summary = se.get_llm_summary(results_df)
        # 4. Visualization & Reporting
        # Trigger LLM Explanation
        analysis_prompt = f"Analyze these pipe bottlenecks: {summary}..."

        if "current_model_idx" not in st.session_state:
            st.session_state.current_model_idx = 0

        for i in range(st.session_state.current_model_idx, len(MODEL_LIST)):
            selected_model = MODEL_LIST[i]
            try:
                result = get_llm_response(analysis_prompt, selected_model)
                st.info(result)
            except (exceptions.ResourceExhausted, Exception):
                st.session_state.current_model_idx = i + 1
                continue


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
    st.header("🤖SDNA Policy Analysis")
    st.write("Demonstrating RFP Task 4: Narrative Framing and Scenario Briefs")

    # Context derived from Denver's actual 80x50 Climate Action Plan
    context = """
    Denver's Climate Action Plan Goals:
    - 80% GHG reduction by 2050.
    - 20% tree canopy coverage in all residential neighborhoods by 2030.
    - Prioritize investments in Equity Priority Areas (Globeville, Elyria-Swansea).
    """

    user_query = st.text_area("Ask SDNA-LLM about policy alignment:",
                              "How does a 15% canopy increase in Globeville support Denver's 2030 goals?")

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
# --- NEW PAGE: FLOOD RESILIENCE SIMULATION ---
elif app_mode == "Flood Resilience Simulation":
    st.title("🌊Denver Flood Resilience Agentic Digital Twin")
    # --- UI IMPLEMENTATION ---
    # st.title("🏙️ Agentic Digital Twin: Natural Language Simulation")
    GOLD_PROMPTS = {
        "Select a Scenario": "",
        "Equity-First Analysis": "Simulate a 3-inch rainfall in May. Focus on neighborhoods with high SVI (Social Vulnerability Index) like Elyria-Swansea. How does low canopy cover exacerbate flood depth there?",
        "Economic Risk Mitigation": "Model a 100-year storm event (5.5 inches) in July 2026. Calculate total property loss for high-value areas like Cherry Creek and suggest where new drainage infrastructure is most cost-effective.",
        "Green Infrastructure Impact": "Compare current flood depths in Five Points during a standard 2-inch rain vs. a hypothetical scenario where canopy cover is increased to 25%. Predict the reduction in surface runoff.",
        "Emergency Response Planning": "Simulate a flash flood in August 2024 with 4.5 inches of rain. Identify which neighborhoods will reach critical inundation levels (>0.5m) first to prioritize evacuation routes."
    }
    st.subheader("📝 Policy Scenario Configuration")
    col_pre, col_edit = st.columns([1, 2])

    with col_pre:
        # Pre-select good prompts based on RFP goals
        selected_template = st.selectbox("RFP Reference Scenarios", list(GOLD_PROMPTS.keys()))

    with col_edit:
        # User can modify the template or write from scratch
        default_text = GOLD_PROMPTS.get(selected_template, "")
        user_query = st.text_area("Refine Scenario Details", value=default_text, height=100,
                                  placeholder="Describe your simulation goal here...")
    # Step 1: User Input
    # user_query = st.text_input("What scenario would you like to explore?",
    #                            placeholder="e.g., 'A 100-year storm in 2026 during the May snowmelt peak'")

    if user_query:
        with st.spinner("Interpreting scenario..."):
            params = parse_scenario_intent(user_query)
            if params:
                # Sync parameters to session state to drive the sliders
                st.session_state.year_val = params.get('year', 2026)
                st.session_state.month_val = params.get('month', 5)
                st.session_state.rain_val = params.get('rain', 3.0)
                st.session_state.target_val = "Property Loss ($)" if params.get('target') == "Loss" else "Flood Depth (m)"

    # Step 2: Controls (Sync'd with AI intent)
    with st.expander("Adjust Parameters & View Prompt Details", expanded=False):
        c1, c2, c3 = st.columns(3)
        year = c1.slider("Year", 2006, 2026, key="year_val", value=st.session_state.get("year_val", 2026))
        month = c2.slider("Month", 1, 12, key="month_val", value=st.session_state.get("month_val", 5))
        rain = c3.slider("Rainfall (Inches)", 0.0, 8.0, key="rain_val", value=st.session_state.get("rain_val", 3.0))
        target_var = st.radio("Focus Metric", ["Flood Depth (m)", "Property Loss ($)"], key="target_val", horizontal=True)

    # Step 3: Run Simulation & Visualization
    st.subheader("Simulated Impacts")
    # REFERENCE PROMPTS (Two-Way Sync)
    scenarios = {
        "Manual Configuration": None,
        "Scenario A: Historical Peak (June 2023)": {"year": 2023, "month": 6, "rain": 4.8},
        "Scenario B: Spring Snowmelt Risk (May 2025)": {"year": 2025, "month": 5, "rain": 2.5},
        "Scenario C: Extreme Monsoonal Flash Flood": {"year": 2026, "month": 8, "rain": 6.2}
    }

    # selected_ref = st.selectbox("Select a Scenario Reference", list(scenarios.keys()))

    # # Update Session State based on selection
    # if scenarios[selected_ref]:
    #     st.session_state.year_val = scenarios[selected_ref]['year']
    #     st.session_state.month_val = scenarios[selected_ref]['month']
    #     st.session_state.rain_val = scenarios[selected_ref]['rain']

    # Parameter Sliders
    # c1, c2, c3 = st.columns(3)
    # year = c1.slider("Simulation Year", 2006, 2026, key="year_val")
    # month = c2.slider("Month", 1, 12, key="month_val")
    # rain = c3.slider("Rainfall Intensity (Inches)", 0.0, 8.0, key="rain_val")

    # Target Selector
    # target_var = st.radio("Primary Target Metric", ["Flood Depth (m)", "Property Loss ($)"], horizontal=True)

    # 4. SIMULATION RESULTS
    # st.subheader("Interactive Impact Analysis")

    # Run Inference for all neighborhoods
    viz_data = []
    for name, p in DENVER_NBHDS.items():
        features = [[year, month, rain, p['SVI'], p['Canopy']]]
        d = model_depth.predict(features)[0]
        l = model_loss.predict(features)[0]
        viz_data.append({"Neighborhood": name, "Depth_m": d, "Loss_USD": l, "SVI": p['SVI'], "Canopy": p['Canopy']})

    res_df = pd.DataFrame(viz_data)

    # col_map, col_text = st.columns([2, 1])

    # with col_map:
    color_col = "Depth_m" if "Depth" in target_var else "Loss_USD"
    fig = px.bar(res_df, x="Neighborhood", y=color_col, color="SVI",
                 title=f"Predicted {target_var} by Neighborhood",
                 color_continuous_scale="Reds")
    st.plotly_chart(fig, use_container_width=True)

    # with col_text:
    st.write("### 🤖 SDNA-LLM Narrative")
    # if st.button("Generate Narrative Report"):
    # prompt = f"""
    #     Analyze this Denver flood scenario:
    #     - Year: {year}, Month: {month}, Rain: {rain} inches.
    #     - Impact: {res_df.to_dict()}
    #     Focus on equity (SVI) and identify which neighborhoods need priority infrastructure in less than 500 words.
    #     """

    # 1. DYNAMIC CONTEXT BUILDING
    # Tell the LLM exactly what the user is looking at.
    focus_area = "economic impact and property loss" if "Property Loss" in target_var else "physical flood depth and hydrology"

    # Convert the top results into a string for the LLM to read
    top_impacts = res_df.sort_values(by="Depth_m" if "Depth" in target_var else "Loss_USD", ascending=False).to_string()

    # 2. THE DYNAMIC PROMPT
    prompt = f"""
    You are a Denver City Policy Advisor. Analyze this SPECIFIC user-generated scenario:
    - FOCUS AREA: {focus_area}
    - PARAMETERS: {year}-{month} with {rain} inches of rainfall.
    - DATA RESULTS:
    {top_impacts}
    
    INSTRUCTIONS:
    1. Explicitly address the '{target_var}' results shown in the chart.
    2. Identify the neighborhood with the highest {target_var}.
    3. Explain the correlation between Canopy ({res_df['Canopy'].mean()}% avg) and the results.
    4. Briefly mention how SVI (Social Vulnerability) creates a 'double burden' for the high-risk areas identified.
    """
    success = True
    for i in range(st.session_state.current_model_idx, len(MODEL_LIST)):
        selected_model = MODEL_LIST[i]

        with st.spinner(f"Analyzing with {selected_model}..."):
            try:
                result = get_llm_response(prompt, selected_model)
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
    # st.info(result)
# --- OVERVIEW ---
# else:
#     st.title("🏙️ Denver Digital Twin: Task 2 & 3 Demo")
#     st.markdown("""
#     ### Technical Alignment:
#     - **Task 2 (Data):** Ingesting live ArcGIS CSVs from Denver Open Data.
#     - **Task 3 (ML):** Running a Tier 1 Random Forest Surrogate for UHI prediction.
#     - **Task 4 (Reporting):** Using Llama 3 for automated "Scenario Briefs".
#     """)
#     if df is not None:
#         st.dataframe(df.head(10))

