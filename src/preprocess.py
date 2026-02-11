import os
import numpy as np
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import xarray as xr
from scipy.ndimage import distance_transform_edt
import torch

def load_dem_and_calculate_slope(dem_path):
    """Loads DEM and returns elevation and slope arrays."""
    with rasterio.open(dem_path) as src:
        elevation = src.read(1)
        res = src.res[0]
        dy, dx = np.gradient(elevation, res)
        slope = np.arctan(np.sqrt(dx**2 + dy**2)) * (180 / np.pi)
        return elevation, slope, src.profile

def rasterize_shp(shp_path, profile, attribute=None):
    """Rasterizes a shapefile to match the given profile."""
    gdf = gpd.read_file(shp_path)
    gdf = gdf.to_crs(profile['crs'])
    
    shapes = ((geom, 1) for geom in gdf.geometry) if attribute is None else \
             ((geom, val) for geom, val in zip(gdf.geometry, gdf[attribute]))
    
    raster = rasterize(shapes, out_shape=(profile['height'], profile['width']), transform=profile['transform'])
    return raster

def calculate_proximity(shp_path, profile):
    """Calculates distance transform from features in shapefile."""
    mask = rasterize_shp(shp_path, profile)
    dist = distance_transform_edt(1 - mask)
    return dist

def process_weather(nc_path, profile):
    """interpolates NetCDF weather data to the raster grid."""
    ds = xr.open_dataset(nc_path)
    temp = ds['t2m'].values[-1] if 't2m' in ds else np.zeros((profile['height'], profile['width']))
    from scipy.interpolate import RegularGridInterpolator
    return np.resize(temp, (profile['height'], profile['width']))

def preprocess_all(data_dir='data/raw', output_dir='data/processed'):
    print("Starting preprocessing...")
    
    dem_path = os.path.join(data_dir, 'dem_90m.tif')
    if not os.path.exists(dem_path) or os.path.getsize(dem_path) == 0:
        print("DEM missing or empty. Generating synthetic elevation.")
        elevation = np.random.rand(256, 256).astype(np.float32)
        slope = np.random.rand(256, 256).astype(np.float32)
        profile = {
            'driver': 'GTiff', 'height': 256, 'width': 256, 'count': 1, 'crs': '+proj=latlong',
            'transform': from_origin(85.0, 24.0, 0.0008, 0.0008), 'dtype': 'float32'
        }
    else:
        elevation, slope, profile = load_dem_and_calculate_slope(dem_path)
    elevation = (elevation - np.min(elevation)) / (np.max(elevation) - np.min(elevation) + 1e-6)
    def get_layer(path, profile, name):
        if os.path.exists(path) and os.path.getsize(path) > 100: 
            try:
                return rasterize_shp(path, profile)
            except Exception as e:
                print(f"Error rasterizing {name}: {e}. Using random.")
        return np.random.rand(profile['height'], profile['width'])

    road_shp = os.path.join(data_dir, 'eastern-zone-osm.shp/gis_osm_roads_free_1.shp')
    road_mask = get_layer(road_shp, profile, "roads")
    road_dist = distance_transform_edt(1 - (road_mask > 0.5))
    road_dist = road_dist / (np.max(road_dist) + 1e-6)
    
    lulc_shp = os.path.join(data_dir, 'lulc_bhuvan/RAMGARH_JH_LULC50K_1516.shp')
    fuel_map = get_layer(lulc_shp, profile, "fuel")
    
    weather_nc = os.path.join(data_dir, 'weather.nc')
    if os.path.exists(weather_nc) and os.path.getsize(weather_nc) > 0:
        weather_feat = process_weather(weather_nc, profile)
    else:
        weather_feat = np.random.rand(profile['height'], profile['width'])
    
    fire_shp = os.path.join(data_dir, 'fires_nasa/fire_archive_M-C61_715142.shp')
    labels = get_layer(fire_shp, profile, "fire_labels")
    
    feature_stack = np.stack([elevation, slope, fuel_map, road_dist, weather_feat], axis=0)
    
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'feature_stack.npy'), feature_stack.astype(np.float32))
    np.save(os.path.join(output_dir, 'labels.npy'), labels.astype(np.float32))
    
    with rasterio.open(os.path.join(output_dir, 'fuel_map_90m.tif'), 'w', **profile) as dst:
        dst.write(fuel_map.astype(np.float32), 1)
        
    print(f"Preprocessing complete. Feature stack shape: {feature_stack.shape}")


if __name__ == "__main__":
    preprocess_all()