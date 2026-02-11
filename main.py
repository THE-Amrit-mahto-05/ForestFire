import os
import torch
import numpy as np
import rasterio
from PIL import Image
from src.model import UNet, get_device
from src.preprocess import preprocess_all
from src.simulation import FireSimulation
from src.utils import save_as_geotiff, generate_fire_gif

def run_pipeline(data_dir='data/raw', output_dir='data/processed', wind_speed=15, wind_dir="North"):
    print(f"Starting Agni-Chakshu Pipeline with Config: Wind {wind_speed}km/h {wind_dir}")
    
    direction_map = {
        "North": (0, -1), "South": (0, 1), "East": (1, 0), "West": (-1, 0),
        "NE": (0.7, -0.7), "SE": (0.7, 0.7), "NW": (-0.7, -0.7), "SW": (-0.7, 0.7)
    }
    base_vec = direction_map.get(wind_dir, (0, 0))
    wind_vector = (base_vec[0] * wind_speed / 15.0, base_vec[1] * wind_speed / 15.0)
    
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
    

    print("Running high-fidelity fire spread simulation with snapshots...")
    risk_map_sim = (risk_map - risk_map.min()) / (risk_map.max() - risk_map.min() + 1e-8)
    fuel_stack = np.load(os.path.join(output_dir, "feature_stack.npy"))
    fuel_map = fuel_stack[2]
    slope_map = fuel_stack[1]
    
    sim = FireSimulation(risk_map_sim, fuel_map, wind_vector=wind_vector, slope_map=slope_map)
    h, w = risk_map.shape
    sim.ignite(h//2, w//2)
    
    hours_list = list(range(1, 13))
    snapshots = sim.run_with_snapshots(hours=hours_list)
    
    os.makedirs("outputs/snapshots", exist_ok=True)
    from src.utils import colorize_simulation_frame_with_burnt
    
    for h in hours_list:
        frame_rgba = colorize_simulation_frame_with_burnt(snapshots[h], sim.fuel_remaining)
        img = Image.fromarray(frame_rgba)
        img.save(f"outputs/snapshots/fire_{h}h.png")
        save_as_geotiff(snapshots[h], profile, f"outputs/maps/fire_spread_{h}h.tif")
    
    sim.reset()
    sim.ignite(h//2, w//2)
    history = []
    for i in range(12 * 4):
        sim.step(dt=0.25)
        if i % 2 == 0:
            history.append((sim.intensity.copy(), sim.fuel_remaining.copy()))
    
    gif_frames = [colorize_simulation_frame_with_burnt(int_map, f_map)[:,:,:3] for int_map, f_map in history]
    generate_fire_gif(gif_frames, "outputs/animations/fire_spread.gif", fps=10)
    
    print("Snapshots and animation saved.")
    print("Pipeline execution complete.")

if __name__ == "__main__":
    import sys
    d_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/raw'
    o_dir = sys.argv[2] if len(sys.argv) > 2 else 'data/processed'
    run_pipeline(data_dir=d_dir, output_dir=o_dir)