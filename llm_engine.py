import streamlit as st
import google.generativeai as genai
from google.api_core import exceptions
import json
import time

# --- CONFIGURATION ---
api_key = st.secrets.get("GEMINI_API_KEY", "None")
genai.configure(api_key=api_key)

# Tiered fallback list as defined in your baseline
MODEL_LIST = [
    'gemini-1.5-flash',       # Primary (Faster)
    'gemini-1.5-flash-8b',    # Backup
    'gemini-1.5-pro'          # High-Power Backup
]

@st.cache_data(ttl=3600)
def get_llm_response(prompt, model_name):
    """Calls the Gemini API with standard throttling."""
    time.sleep(2) # Prevent rapid-fire quota exhaustion
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except exceptions.ResourceExhausted:
        raise exceptions.ResourceExhausted("Rate limit reached.")
    except Exception as e:
        return f"Error: {str(e)}"

def parse_scenario_intent(user_input):
    """Agentic Parser: Turns natural language into slider values."""
    prompt = f"""
    Convert this user request into a JSON scenario object for a flood simulator.
    User Request: "{user_input}"
    Rules:
    - Year: 2006 to 2026.
    - Month: 1 to 12.
    - Rainfall_In: 0.0 to 8.0.
    - Target: "Depth" or "Loss".
    Return ONLY JSON: {{"year": 2026, "month": 6, "rain": 4.5, "target": "Loss"}}
    """

    # Implementation of your fallback loop
    if "current_model_idx" not in st.session_state:
        st.session_state.current_model_idx = 0

    for i in range(st.session_state.current_model_idx, len(MODEL_LIST)):
        selected_model = MODEL_LIST[i]
        try:
            result = get_llm_response(prompt, selected_model)
            # Clean and return JSON
            clean_json = result.strip().replace('```json', '').replace('```', '')
            return json.loads(clean_json)
        except (exceptions.ResourceExhausted, Exception):
            st.session_state.current_model_idx = i + 1
            continue
    return None