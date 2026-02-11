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

def generate_fire_gif(frames, output_path, fps=5):
    """Converts a list of fire mask frames into a GIF."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig, ax = plt.subplots()
    ims = []
    for frame in frames:
        im = ax.imshow(frame, animated=True, cmap='hot')
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=1000/fps, blit=True)
    ani.save(output_path, writer='pillow')
    plt.close()

def normalize(array):
    """Standard min-max normalization."""
    return (array - array.min()) / (array.max() - array.min() + 1e-8)

def colorize_risk_map(risk_map):
    """Converts a 0-1 probability map into an RGBA heatmap."""

    heatmap = (risk_map * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGBA)
    
    alpha = np.where(risk_map < 0.1, 0, 180).astype(np.uint8)
    colored[:, :, 3] = alpha
    return colored

def array_to_png_base64(array):
    """Encodes an RGBA array to a base64 PNG string."""
    from io import BytesIO
    import base64
    img = Image.fromarray(array)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()