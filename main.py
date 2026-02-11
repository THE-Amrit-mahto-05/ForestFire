import os
import torch
import numpy as np
import rasterio
from src.model import UNet, get_device
from src.preprocess import preprocess_all
from src.simulation import FireSimulation
from src.utils import save_as_geotiff, generate_fire_gif

def run_pipeline(data_dir='data/raw', output_dir='data/processed'):
    print("Starting Agni-Chakshu Pipeline")
    
    feature_stack_path = os.path.join(output_dir, "feature_stack.npy")
    if not os.path.exists(feature_stack_path):
        preprocess_all(data_dir=data_dir, output_dir=output_dir)
    else:
        print(f"Data already processed at {output_dir}. Skipping.")

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
    features = np.load(os.path.join(output_dir, "feature_stack.npy"))
    input_tensor = torch.from_numpy(features).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(input_tensor)
    
    risk_map = prediction.squeeze().cpu().numpy()
    
    with rasterio.open("data/raw/dem_90m.tif") as src:
        profile = src.profile
    

    risk_map_sim = (risk_map - risk_map.min()) / (risk_map.max() - risk_map.min() + 1e-8)
    fuel_stack = np.load(os.path.join(output_dir, "feature_stack.npy"))
    fuel_map = fuel_stack[2] if fuel_stack.ndim == 3 else fuel_stack
    
    sim = FireSimulation(risk_map_sim, fuel_map, wind_vector=(1, 1))
    h, w = risk_map.shape
    sim.ignite(h//2, w//2)
    
    hourly_states = sim.run()
    
    from src.utils import colorize_simulation_frame
    frames = [colorize_simulation_frame(s)[:,:,:3] for s in hourly_states.values()]
    generate_fire_gif(frames, "outputs/animations/fire_spread.gif")
    
    final_state = hourly_states[12]
    save_as_geotiff(final_state, profile, "outputs/maps/fire_spread_12h.tif")
    
    print("Simulation outputs saved.")

    print("Pipeline execution complete.")

if __name__ == "__main__":
    import sys
    d_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/raw'
    o_dir = sys.argv[2] if len(sys.argv) > 2 else 'data/processed'
    run_pipeline(data_dir=d_dir, output_dir=o_dir)