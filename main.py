import torch
import numpy as np
from src.model import UNet

def predict_fire_risk():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    
    features = np.load('data/processed/feature_stack.npy')
    input_tensor = torch.from_numpy(features).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor)
    
    risk_map = prediction.squeeze().cpu().numpy()
    np.save('outputs/maps/latest_risk.npy', risk_map)
    print("Prediction complete. Map saved in outputs/maps/")

if __name__ == "__main__":
    predict_fire_risk()