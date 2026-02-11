import rasterio
import numpy as np

def save_prediction_as_tif(prediction, reference_tif, output_path):
    with rasterio.open(reference_tif) as src:
        meta = src.meta.copy()
        meta.update(dtype='float32', count=1)
        
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(prediction.astype('float32'), 1)