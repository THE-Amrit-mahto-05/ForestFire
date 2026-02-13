import numpy as np
from scipy.ndimage import convolve

class FireSimulation:
    def __init__(self, risk_map, fuel_map, wind_vector=(1, 1), slope_map=None):
        """
        Advanced Cellular Automata for Dynamic Fire Spread.
        intensity: 0.0=Unburnt, 0.1-0.3=Cooling/Charcoal, 0.4-0.7=Active, 0.8-1.0=Peak
        """
        self.risk_map = risk_map
        self.fuel_map = fuel_map.copy()
        self.slope_map = slope_map if slope_map is not None else np.zeros_like(risk_map)
        self.wind_vector = np.array(wind_vector)
        self.height, self.width = risk_map.shape
        self.reset()

    def reset(self):
        self.intensity = np.zeros((self.height, self.width), dtype=np.float32)
        self.fuel_remaining = np.ones((self.height, self.width), dtype=np.float32)
        self.age = np.zeros((self.height, self.width), dtype=np.float32)

    def ignite(self, y, x, radius=2):
        """Ignites a starting area."""
        y_min, y_max = max(0, y-radius), min(self.height, y+radius)
        x_min, x_max = max(0, x-radius), min(self.width, x+radius)
        self.intensity[y_min:y_max, x_min:x_max] = 0.8
        self.age[y_min:y_max, x_min:x_max] = 0.1

    def run_with_snapshots(self, hours=[1, 2, 3, 6, 12], steps_per_hour=4):
        """Runs simulation and returns specific temporal snapshots."""
        snapshots = {}
        total_steps = max(hours) * steps_per_hour
        dt = 1.0 / steps_per_hour
        
        current_step = 0
        for h in sorted(hours):
            target_step = h * steps_per_hour
            while current_step < target_step:
                self.step(dt=dt)
                current_step += 1
            snapshots[h] = self.intensity.copy()
            
        return snapshots

    def step(self, dt=0.25):
        """Advances simulation by dt hours with multi-stage physics (Vectorized)."""
        # 1. Spread Logic: Vectorized for Efficiency
        potential_mask = (self.fuel_remaining > 0.1) & (self.intensity < 0.4)
        if not np.any(self.intensity > 0.4):
            return # No active fire to spread
            
        new_intensity = self.intensity.copy()
        
        # Shifted arrays for 8 neighbors
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0: continue
                
                # Shifted intensity (heat source)
                # Note: We shift the 'source' to the 'target'
                shifted_intensity = np.zeros_like(self.intensity)
                if dy == -1:    # Source is north, target is south
                    s_y, t_y = slice(0, -1), slice(1, None)
                elif dy == 1:   # Source is south, target is north
                    s_y, t_y = slice(1, None), slice(0, -1)
                else:
                    s_y, t_y = slice(None), slice(None)
                    
                if dx == -1:    # Source is west, target is east
                    s_x, t_x = slice(0, -1), slice(1, None)
                elif dx == 1:   # Source is east, target is west
                    s_x, t_x = slice(1, None), slice(0, -1)
                else:
                    s_x, t_x = slice(None), slice(None)
                
                shifted_intensity[t_y, t_x] = self.intensity[s_y, s_x]
                
                # Probability calculation for this specific direction
                heat = shifted_intensity
                wind_eff = dx * self.wind_vector[0] + dy * self.wind_vector[1]
                
                # Approximate slope effect (simplified vectorization)
                prob = (heat * self.risk_map * self.fuel_map)
                prob *= (1.0 + 0.4 * wind_eff) # Directional bias
                
                # Update candidates
                # Only apply spread where target is ignitable
                ignite_mask = potential_mask & (np.random.rand(*self.intensity.shape) < prob * dt * 3.5)
                new_intensity[ignite_mask] = np.maximum(new_intensity[ignite_mask], 0.5)

        # 2. Life Cycle & Consumption
        # Increment age for burning cells
        self.age[self.intensity > 0.1] += dt
        
        # Heat consumes fuel
        consumption = self.intensity * 0.3 * dt
        self.fuel_remaining = np.clip(self.fuel_remaining - consumption, 0, 1)
        
        # Intensity evolves: Peak -> Cooling -> Charcoal -> Out
        # active cells (>0.4)
        peak_mask = (self.intensity >= 0.4) & (self.fuel_remaining > 0.2)
        cooling_mask = (self.intensity > 0.1) & (self.fuel_remaining <= 0.2)
        charcoal_mask = (self.intensity > 0.0) & (self.fuel_remaining <= 0.05)

        # Increase intensity if fuel is plenty
        new_intensity[peak_mask] = np.clip(new_intensity[peak_mask] + 0.1 * dt, 0.4, 1.0)
        # Drop intensity as fuel runs out (cooling phase)
        new_intensity[cooling_mask] = np.clip(new_intensity[cooling_mask] - 0.4 * dt, 0.1, 0.4)
        # Final charcoal phase
        new_intensity[charcoal_mask] = np.clip(new_intensity[charcoal_mask] - 0.2 * dt, 0.0, 0.2)
        
        # 3. Global update
        self.intensity = new_intensity
        self.intensity[self.fuel_remaining < 0.01] = np.clip(self.intensity[self.fuel_remaining < 0.01], 0, 0.1) # charcoal footprint

