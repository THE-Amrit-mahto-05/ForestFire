import os
import torch
import numpy as np
import rasterio
from src.model import UNet, get_device
from src.preprocess import preprocess_all
from src.simulation import FireSimulation
from src.utils import save_as_geotiff, generate_fire_gif

def run_pipeline():
    print("Starting Agni-Chakshu Pipeline")
    
    if not os.path.exists("data/processed/feature_stack.npy"):
        preprocess_all()
    else:
        print("Data already processed. Skipping.")

    device = get_device()
    print(f"Using device: {device}")
    
    model = UNet(in_channels=5).to(device)
    model_path = "models/unet_fire_model.pth"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded trained model weights.")
    else:
        print("No trained weights found. Using random initialization for demonstration.")

    model.eval()
    features = np.load("data/processed/feature_stack.npy")
    input_tensor = torch.from_numpy(features).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(input_tensor)
    
    risk_map = prediction.squeeze().cpu().numpy()
    
    with rasterio.open("data/raw/dem_90m.tif") as src:
        profile = src.profile
    
    save_as_geotiff(risk_map, profile, "outputs/maps/latest_risk.tif")
    print("Risk map saved to outputs/maps/latest_risk.tif")

    print("Running fire spread simulation...")

    fuel_map = np.load("data/processed/feature_stack.npy")[2] 
    
    sim = FireSimulation(risk_map, fuel_map, wind_vector=(1, 1))
    h, w = risk_map.shape
    sim.ignite(h//2, w//2)
    
    frames = sim.run(steps=12)
    generate_fire_gif(frames, "outputs/animations/fire_spread.gif")
    print("Simulation GIF saved to outputs/animations/fire_spread.gif")

    print("Pipeline execution complete.")

if __name__ == "__main__":
    import sys
    # Allow running with custom dirs for demo
    d_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/raw'
    o_dir = sys.argv[2] if len(sys.argv) > 2 else 'data/processed'
    run_pipeline(data_dir=d_dir, output_dir=o_dir)