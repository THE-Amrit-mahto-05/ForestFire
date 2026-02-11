import sys
import os
sys.path.append(os.getcwd())
try:
    from src.utils import colorize_terrain_map, colorize_fuel_map
    print("SUCCESS: Functions imported correctly.")
except ImportError as e:
    print(f"FAILURE: {e}")
    import src.utils
    print(f"Available attributes in src.utils: {dir(src.utils)}")
