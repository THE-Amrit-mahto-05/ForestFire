import numpy as np
import rasterio
from rasterio.transform import from_origin
import cv2
from PIL import Image

def save_as_geotiff(data, profile, output_path):
    """Saves a numpy array as a GeoTIFF using the reference profile."""
    if data.ndim == 3:
        data = data[0]
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data.astype(np.float32), 1)

def generate_fire_gif(frames, output_path, fps=10):
    """Converts a list of fire mask frames into a GIF."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    fig.patch.set_facecolor('black')
    ax.axis('off')
    
    ims = []
    for frame in frames:
        if frame.ndim == 2:
            frame = colorize_simulation_heatmap(frame)
        
        im = ax.imshow(frame, animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=1000/fps, blit=True)
    ani.save(output_path, writer='pillow')
    plt.close()

def normalize(array):
    """Standard min-max normalization."""
    return (array - array.min()) / (array.max() - array.min() + 1e-8)

def colorize_risk_map(risk_map):
    """Pro-looking Fire Risk Map (Red/Hot)."""
    heatmap = (risk_map * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGBA)
    
    alpha = np.where(risk_map < 0.15, 0, 200).astype(np.uint8)
    colored[:, :, 3] = alpha
    return colored

def colorize_simulation_heatmap(intensity):
    """Colors intensity for map visibility: High-contrast fire colors."""
    norm_int = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-8)
    heatmap = (norm_int * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    
    mask = (intensity > 0.01)
    colored[mask, 0] = np.maximum(colored[mask, 0], 150) 
    return colored

def colorize_simulation_frame_with_burnt(intensity, fuel_remaining):
    """
    Cinematic Fire Visualization: 
    - Forest Green Background
    - Heat Glow (Gaussian Blur)
    - Bright Fire Fronts
    - Charcoal Residue
    - Subtle Smoke/Ash Footprint
    """
    h, w = intensity.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    
    rgba[:, :] = [10, 20, 10, 255] 
    
    burnt_mask = (fuel_remaining < 0.98)
    if np.any(burnt_mask):
        ash_factor = (1.0 - fuel_remaining[burnt_mask])
        rgba[burnt_mask, 0] = 30 + ash_factor * 30
        rgba[burnt_mask, 1] = 30 + ash_factor * 30
        rgba[burnt_mask, 2] = 32 + ash_factor * 30 
        rgba[burnt_mask, 3] = 255
    
    if np.any(intensity > 0.1):
        glow_map = cv2.GaussianBlur(intensity, (15, 15), 0)
        glow_mask = (glow_map > 0.05)
        rgba[glow_mask, 0] = np.clip(rgba[glow_mask, 0] + glow_map[glow_mask] * 180, 0, 255)
        rgba[glow_mask, 1] = np.clip(rgba[glow_mask, 1] + glow_map[glow_mask] * 50, 0, 255)
        rgba[glow_mask, 2] = np.clip(rgba[glow_mask, 2] + glow_map[glow_mask] * 10, 0, 255)

    fire_mask = (intensity > 0.1)
    if np.any(fire_mask):
        sub_int = intensity[fire_mask]
        r = np.clip(sub_int * 255 * 1.5, 150, 255)
        g = np.clip(sub_int * 255 * 0.8, 50, 255)
        b = np.clip(sub_int * 255 * 0.2, 0, 100)
        rgba[fire_mask, 0] = r
        rgba[fire_mask, 1] = g
        rgba[fire_mask, 2] = b
        rgba[fire_mask, 3] = 255
        
        hot_core = (intensity > 0.75)
        rgba[hot_core, 0:3] = [255, 250, 200]
    
    return rgba

def array_to_png_base64(array):
    """Encodes an RGBA array to a base64 PNG string."""
    from io import BytesIO
    import base64
    img = Image.fromarray(array)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def colorize_terrain_map(elevation):
    """Topographic visualization of elevation/slope."""
    norm = (elevation - elevation.min()) / (elevation.max() - elevation.min() + 1e-6)
    heatmap = (norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_BONE)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGBA)
    colored[:, :, 3] = 180
    return colored

def colorize_fuel_map(fuel_map):
    """LULC visualization: Forests, vegetation, etc."""
    heatmap = (fuel_map * 127).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_SUMMER)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGBA)
    colored[fuel_map == 0, 3] = 0
    colored[fuel_map > 0, 3] = 160
    return colored