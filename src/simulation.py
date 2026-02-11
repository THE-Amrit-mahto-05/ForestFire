import numpy as np

class FireSimulation:
    def __init__(self, risk_map, fuel_map, wind_vector=(1, 1)):
        """
        Refined Cellular Automata for ISRO PS1.
        States: 0=Unburnt, 1=Burning, 2=Burnt
        """
        self.risk_map = risk_map
        self.fuel_map = fuel_map
        self.wind_vector = np.array(wind_vector)
        self.height, self.width = risk_map.shape
        self.reset()

    def reset(self):
        self.state = np.zeros((self.height, self.width), dtype=np.uint8)

    def ignite(self, y, x):
        """Ignites a larger cluster for visibility."""
        r = 3
        self.state[max(0, y-r):min(self.height, y+r), max(0, x-r):min(self.width, x+r)] = 1

    def run(self):
        """Runs the simulation and returns snapshots for 1, 2, 3, 6, 12 hours."""
        history = {0: self.state.copy()}
        checkpoints = [1, 2, 3, 6, 12]
        
        current_hour = 0
        while current_hour < 12:
            current_hour += 1
            self.step()
            if current_hour in checkpoints:
                history[current_hour] = self.state.copy()
        
        return history

    def step(self):
        """Advances simulation by 1 hour. Aggressive spread for demo."""
        new_state = self.state.copy()
        is_burning = (self.state == 1)
        
        # Check 8-neighbors
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0: continue
                
                shifted = np.roll(np.roll(is_burning, dy, axis=0), dx, axis=1)
                
                w_norm = np.linalg.norm(self.wind_vector) + 1e-6
                wind_eff = np.dot([dx, dy], self.wind_vector) / w_norm
                
                # Boosted spread probability for demo visibility
                spread_prob = (self.risk_map + self.fuel_map) * (0.8 + 0.4 * wind_eff)
                
                ignite_mask = (np.random.rand(self.height, self.width) < spread_prob * 0.9)
                new_state[(self.state == 0) & shifted & ignite_mask] = 1
        
        # Current burning becomes burnt
        new_state[is_burning] = 2
        self.state = new_state
