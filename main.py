from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random
import time

app = FastAPI()

# Enable CORS so your GitHub Pages frontend can talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace '*' with your GitHub Pages URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initial "Plant" State
state = {
    "inlet_flow": 1200,      # m3/hr
    "alum_dosage": 12.5,     # mg/L
}

class SimulationInput(BaseModel):
    proposed_dosage: float

@app.get("/live-scada")
def get_live_data():
    # Simulate slight noise in sensor readings (The "Live" feel)
    return {
        "timestamp": time.time(),
        "tags": {
            "Flow_Rate": state["inlet_flow"] + random.uniform(-10, 10),
            "Current_Dosage": state["alum_dosage"],
            # Base turbidity is 45, fluctuates slightly
            "Inlet_Turbidity": 45 + random.uniform(-2, 2),
            # Effluent is good (0.8) if dose is near 12.5
            "Effluent_Quality": 0.8 + abs(state["alum_dosage"] - 12.5) * 0.1 + random.uniform(-0.05, 0.05)
        }
    }

@app.post("/simulate")
def run_simulation(inputs: SimulationInput):
    # This represents the "Advanced Simulation Engine"
    # Logic: Deviation from optimal dose (12.5) causes turbidity spike
    optimal_dose = 12.5
    base_turbidity = 0.8

    dose_diff = abs(optimal_dose - inputs.proposed_dosage)
    # Simulation formula: Quality degrades exponentially with bad dosage
    predicted_quality = base_turbidity + (dose_diff ** 1.5) * 0.2

    # Financial Impact: Alum costs $0.50 per mg/L per m3
    daily_cost = state["inlet_flow"] * 24 * inputs.proposed_dosage * 0.50

    return {
        "scenario": "Dosage Optimization",
        "predicted_effluent_turbidity": round(predicted_quality, 2),
        "daily_chemical_cost": round(daily_cost, 2),
        "status": "Safe" if predicted_quality < 1.5 else "CRITICAL VIOLATION"
    }