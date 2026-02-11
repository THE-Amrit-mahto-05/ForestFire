import numpy as np
from scipy.ndimage import convolve

class FireSimulation:
    def __init__(self, risk_map, fuel_map, wind_vector=(1, 0)):
        """
        Cellular Automata simulation for Fire Spread.
        risk_map: 2D array of ignition probability (0-1).
        fuel_map: 2D array of fuel availability (0-1).
        wind_vector: (dx, dy) direction of wind.
        """
        self.risk_map = risk_map
        self.fuel_map = fuel_map
        self.wind_vector = np.array(wind_vector)
        self.height, self.width = risk_map.shape
        self.reset()

    def reset(self):
        """Resets the simulation state (fire mask)."""
        self.fire_mask = np.zeros((self.height, self.width), dtype=np.float32)

    def ignite(self, y, x):
        """Manually ignites a point."""
        self.fire_mask[y, x] = 1.0

    def step(self):
        """Executes one iteration of fire spread."""
        kernel = np.array([
            [0.1, 0.2, 0.1],
            [0.2, 1.0, 0.2],
            [0.1, 0.2, 0.1]
        ])
        
        neighbor_sum = convolve(self.fire_mask, kernel, mode='constant', cval=0)
        
        dx, dy = self.wind_vector
        shifted_neighbors = np.roll(neighbor_sum, shift=(int(dy), int(dx)), axis=(0, 1))
        
        spread_chance = shifted_neighbors * self.risk_map * self.fuel_map
        new_fire = (spread_chance > 0.5).astype(np.float32)

        self.fire_mask = np.clip(self.fire_mask + new_fire, 0, 1)
        self.fuel_map = np.clip(self.fuel_map - (self.fire_mask * 0.05), 0, 1)
        
        return self.fire_mask.copy()

    def run(self, steps=12):
        """Runs the simulation for N steps and returns frames."""
        frames = [self.fire_mask.copy()]
        for _ in range(steps):
            frames.append(self.step())
        return frames
