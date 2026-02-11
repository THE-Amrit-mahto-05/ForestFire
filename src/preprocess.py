import geopandas as gpd
import rasterio
from rasterio import features
import numpy as np
import xarray as xr
from scipy.ndimage import distance_transform_edt

def preprocess_all():
    with rasterio.open('data/raw/dem_90m.tif') as src:
        dem_arr = src.read(1)
        meta = src.meta
        transform = src.transform
        out_shape = src.shape
        crs = src.crs

    print(f"Master Grid Initialized: {out_shape} at 90m resolution.")

    lulc_gdf = gpd.read_file('data/raw/lulc_bhuvan/RAMGARH_JH_LULC50K_1516.shp').to_crs(crs)
    if 'LU_CODE' not in lulc_gdf.columns:
        lulc_gdf['LU_CODE'] = lulc_gdf['CLASS'].astype('category').cat.codes
    
    shapes = ((geom, val) for geom, val in zip(lulc_gdf.geometry, lulc_gdf.LU_CODE))
    fuel_layer = features.rasterize(shapes=shapes, out_shape=out_shape, transform=transform)

    roads_gdf = gpd.read_file('data/raw/eastern-zone-osm.shp/gis_osm_roads_free_1.shp').to_crs(crs)
    road_mask = features.rasterize(
        ((geom, 1) for geom in roads_gdf.geometry),
        out_shape=out_shape, transform=transform, fill=0
    )
    dist_to_road = distance_transform_edt(road_mask == 0) 

    fires_gdf = gpd.read_file('data/raw/fires_nasa/fire_archive_M-C61_715142.shp').to_crs(crs)
    fire_labels = features.rasterize(
        ((geom, 1) for geom in fires_gdf.geometry),
        out_shape=out_shape, transform=transform, fill=0
    )

    def normalize(array):
        return (array - array.min()) / (array.max() - array.min() + 1e-8)

    feature_stack = np.stack([
        normalize(dem_arr),
        normalize(fuel_layer),
        normalize(dist_to_road)
    ], axis=0)

    np.save('data/processed/feature_stack.npy', feature_stack.astype(np.float32))
    np.save('data/processed/labels.npy', fire_labels.astype(np.float32))
    
    print("Pre-processing Complete! Data saved to data/processed/")

if __name__ == "__main__":
    preprocess_all()