import rasterio
from rasterio.enums import Resampling
import numpy as np

def get_master_template(dem_path):
    """
    Reads the 30m DEM to establish the 'Golden Standard' 
    for CRS, transform, and dimensions.
    """
    with rasterio.open(dem_path) as src:
        master_meta = src.meta.copy()
        master_shape = src.shape  
        master_transform = src.transform
        master_crs = src.crs
        
    print(f"Master Grid Established: {master_shape} at 30m resolution.")
    return master_meta

def resample_to_master(input_path, master_meta, output_path):
    """
    Resamples any .tif (like LULC or Weather) to match the 30m Master Grid.
    """
    with rasterio.open(input_path) as src:
        data = np.empty((master_meta['count'], master_meta['height'], master_meta['width']), 
                         dtype=master_meta['dtype'])
        rasterio.warp.reproject(
            source=rasterio.band(src, 1),
            destination=data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=master_meta['transform'],
            dst_crs=master_meta['crs'],
            resampling=Resampling.bilinear
        )
        
        with rasterio.open(output_path, 'w', **master_meta) as dst:
            dst.write(data)
    print(f"Saved resampled file to: {output_path}")