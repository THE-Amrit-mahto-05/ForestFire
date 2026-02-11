import numpy as np
import rasterio
from rasterio.transform import from_origin
import xarray as xr
import os

def generate_synthetic_data(data_dir='data/raw'):
    print("🛠 Generating synthetic data for demonstration...")
    os.makedirs(data_dir, exist_ok=True)

    dem_path = os.path.join(data_dir, 'dem_90m.tif')
    res = 90.0
    size = 256
    y, x = np.mgrid[-size//2:size//2, -size//2:size//2]
    elevation = 500 + 200 * np.exp(-(x**2 + y**2) / (size**2 / 10))
    
    transform = from_origin(85.0, 24.0, res/111000, res/111000)
    new_dataset = rasterio.open(
        dem_path, 'w', driver='GTiff',
        height=size, width=size,
        count=1, dtype=elevation.dtype,
        crs='+proj=latlong',
        transform=transform,
    )
    new_dataset.write(elevation, 1)
    new_dataset.close()
    print(f"Created {dem_path}")

    weather_path = os.path.join(data_dir, 'weather.nc')
    temp = 30 + 5 * np.random.randn(1, size, size)
    ds = xr.Dataset(
        {
            "t2m": (("time", "y", "x"), temp),
        },
        coords={
            "time": [0],
            "y": range(size),
            "x": range(size),
        },
    )
    ds.to_netcdf(weather_path)
    print(f"Created {weather_path}")
    print("Mocking complete. Ready for pipeline test.")

if __name__ == "__main__":
    generate_synthetic_data()
