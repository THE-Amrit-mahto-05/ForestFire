from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import os
from src.model import UNet, get_device

app = FastAPI(title="Agni-Chakshu API")

class PredictionRequest(BaseModel):
    region_id: str = "jharkhand_central"

@app.get("/")
async def root():
    return {"message": "Agni-Chakshu API is online", "system": "Jharkhand Forest Fire Intelligence"}

@app.post("/predict")
async def predict_risk(request: PredictionRequest):
    device = get_device()
    
    model = UNet(in_channels=5).to(device)
    model_path = "models/unet_fire_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.eval()
    
    if not os.path.exists("data/processed/feature_stack.npy"):
        raise HTTPException(status_code=404, detail="Processed data not found. Run preprocessing first.")
        
    features = np.load("data/processed/feature_stack.npy")
    input_tensor = torch.from_numpy(features).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(input_tensor)
    
    risk_map = prediction.squeeze().cpu().numpy()
    
    output_path = "outputs/maps/latest_risk.npy"
    np.save(output_path, risk_map)
    
    return {
        "status": "success",
        "risk_mean": float(np.mean(risk_map)),
        "risk_max": float(np.max(risk_map)),
        "output_path": output_path
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
