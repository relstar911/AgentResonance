import numpy as np
import random
from sklearn.metrics import mutual_info_score
from collections import deque, Counter
from scipy.stats import pearsonr
import pandas as pd

# GridWorld-Konfiguration
grid_size = 10

def generate_grid(grid_size=10, resources=10, dangers=5, goals=1):
    """Generate a grid world with resources, dangers, and goals"""
    grid = np.zeros((grid_size, grid_size))
    
    # Place resources
    for _ in range(resources):
        x, y = np.random.randint(0, grid_size), np.random.randint(0, grid_size)
        grid[x, y] = 1
    
    # Place dangers
    for _ in range(dangers):
        x, y = np.random.randint(0, grid_size), np.random.randint(0, grid_size)
        grid[x, y] = -1
    
    # Place goals
    for _ in range(goals):
        x, y = np.random.randint(0, grid_size), np.random.randint(0, grid_size)
        grid[x, y] = 2
        
    return grid

class ConsciousAgent:
    def __init__(self, name, start_x=None, start_y=None, grid_size=10):
        self.name = name
        self.x = start_x if start_x is not None else random.randint(0, grid_size - 1)
        self.y = start_y if start_y is not None else random.randint(0, grid_size - 1)
        self.energy = 100
        self.prev_action = None
        self.state_memory = []
        self.activity_log = []
        self.decision_history = deque(maxlen=10)
        self.last_position = (self.x, self.y)
        self.bond_strength = 0.0  # Erinnerung an Beziehung
        self.last_b = 0.0
        self.path_history = [(self.x, self.y)]  # Track path for visualization
        self.grid_size = grid_size

    def sense(self, grid, other_agent):
        view = grid[max(0, self.x-1):min(self.grid_size, self.x+2),
                    max(0, self.y-1):min(self.grid_size, self.y+2)]
        sees_other = abs(self.x - other_agent.x) <= 3 and abs(self.y - other_agent.y) <= 3
        flat_view = view.flatten()
        self.activity_log.append(flat_view)

        if sees_other:
            self.bond_strength += 0.01
        else:
            self.bond_strength -= 0.005
        self.bond_strength = max(0.0, min(1.0, self.bond_strength))

        return view, sees_other

    def self_model(self):
        predicted_x = self.x
        predicted_y = self.y
        if self.prev_action == 'UP': predicted_y -= 1
        elif self.prev_action == 'DOWN': predicted_y += 1
        elif self.prev_action == 'LEFT': predicted_x -= 1
        elif self.prev_action == 'RIGHT': predicted_x += 1
        prediction = (predicted_x, predicted_y)
        self.state_memory.append(((self.x, self.y), prediction))
        return prediction

    def move_towards(self, other):
        dx = other.x - self.x
        dy = other.y - self.y
        if abs(dx) > abs(dy):
            return 'RIGHT' if dx > 0 else 'LEFT'
        elif dy != 0:
            return 'DOWN' if dy > 0 else 'UP'
        return 'STAY'

    def decide(self, sees_other, other_agent):
        actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']
        if sees_other or self.bond_strength > 0.2:
            if random.random() < self.bond_strength:
                action = self.move_towards(other_agent)
            else:
                action = random.choices(actions, weights=[1,1,1,1,0.5])[0]
        else:
            action = random.choices(actions, weights=[1,1,1,1,0.5])[0]
        self.prev_action = action
        self.decision_history.append(action)
        return action

    def act(self, action):
        self.last_position = (self.x, self.y)
        if action == 'UP' and self.y > 0: self.y -= 1
        elif action == 'DOWN' and self.y < self.grid_size - 1: self.y += 1
        elif action == 'LEFT' and self.x > 0: self.x -= 1
        elif action == 'RIGHT' and self.x < self.grid_size - 1: self.x += 1
        self.energy -= 1
        self.path_history.append((self.x, self.y))

    def get_state(self):
        return (self.x, self.y, self.energy)

    def calculate_phi(self):
        if len(self.activity_log) < 2:
            return 1.0
        cut_1 = self.activity_log[-2]
        cut_2 = self.activity_log[-1]
        min_len = min(len(cut_1), len(cut_2))
        cut_1 = cut_1[:min_len]
        cut_2 = cut_2[:min_len]
        mi_full = mutual_info_score(cut_1, cut_1)
        mi_cut = mutual_info_score(cut_1, cut_2)
        phi = 1.0 - (mi_cut / mi_full) if mi_full != 0 else 0.0
        return max(0.0, min(1.0, phi))

    def calculate_delta(self):
        if not self.decision_history:
            return 0.5
        counter = Counter(self.decision_history)
        probs = [v / len(self.decision_history) for v in counter.values()]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        max_entropy = np.log2(len(set(['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY'])))
        return min(1.0, entropy / max_entropy)

    def set_last_b(self, b):
        self.last_b = b

def calculate_resonance(a_b, b_b):
    return 1.0 - abs(a_b - b_b)

def run_simulation(steps=100, grid_size=10, resources=10, dangers=5, goals=1, 
                  start_pos_a=(None, None), start_pos_b=(None, None)):
    """Run simulation with given parameters and return log data"""
    grid = generate_grid(grid_size, resources, dangers, goals)
    agent_a = ConsciousAgent("A", start_x=start_pos_a[0], start_y=start_pos_a[1], grid_size=grid_size)
    agent_b = ConsciousAgent("B", start_x=start_pos_b[0], start_y=start_pos_b[1], grid_size=grid_size)
    log = []
    b_values_a = []
    b_values_b = []
    
    grid_snapshots = [grid.copy()]  # Initial grid
    agent_positions = [[(agent_a.x, agent_a.y), (agent_b.x, agent_b.y)]]  # Initial positions

    for t in range(steps):
        view_a, sees_b = agent_a.sense(grid, agent_b)
        view_b, sees_a = agent_b.sense(grid, agent_a)

        pred_a = agent_a.self_model()
        pred_b = agent_b.self_model()

        act_a = agent_a.decide(sees_b, agent_b)
        act_b = agent_b.decide(sees_a, agent_a)

        agent_a.act(act_a)
        agent_b.act(act_b)

        state_a = agent_a.get_state()
        state_b = agent_b.get_state()

        sigma_a = 1.0 - (np.linalg.norm(np.array(state_a[:2]) - np.array(pred_a)) / grid_size)
        sigma_b = 1.0 - (np.linalg.norm(np.array(state_b[:2]) - np.array(pred_b)) / grid_size)

        sigma_a = max(0.0, min(1.0, sigma_a))
        sigma_b = max(0.0, min(1.0, sigma_b))

        phi_a = agent_a.calculate_phi()
        phi_b = agent_b.calculate_phi()

        delta_a = agent_a.calculate_delta()
        delta_b = agent_b.calculate_delta()

        b_a = phi_a * sigma_a * delta_a
        b_b = phi_b * sigma_b * delta_b

        agent_a.set_last_b(b_a)
        agent_b.set_last_b(b_b)

        b_values_a.append(b_a)
        b_values_b.append(b_b)

        resonance = calculate_resonance(b_a, b_b)
        
        # Record positions after movement
        agent_positions.append([(agent_a.x, agent_a.y), (agent_b.x, agent_b.y)])
        
        # Store current grid state (could be modified to show dynamic environment changes)
        grid_snapshots.append(grid.copy())

        log.append({
            't': t,
            'A_x': agent_a.x, 'A_y': agent_a.y,
            'B_x': agent_b.x, 'B_y': agent_b.y,
            'A_energy': agent_a.energy, 'B_energy': agent_b.energy,
            'A_phi': phi_a, 'A_sigma': sigma_a, 'A_delta': delta_a, 'A_B': b_a,
            'B_phi': phi_b, 'B_sigma': sigma_b, 'B_delta': delta_b, 'B_B': b_b,
            'A_sees_B': sees_b, 'B_sees_A': sees_a,
            'A_bond': agent_a.bond_strength, 'B_bond': agent_b.bond_strength,
            'resonance': resonance,
            'A_action': act_a, 'B_action': act_b
        })

    relationship_score = pearsonr(b_values_a, b_values_b)[0] if len(b_values_a) > 1 else 0.0
    
    # Convert log to DataFrame for easier analysis
    df_log = pd.DataFrame(log)
    
    simulation_results = {
        'grid': grid,
        'log': df_log,
        'agent_a': agent_a,
        'agent_b': agent_b,
        'relationship_score': relationship_score,
        'grid_snapshots': grid_snapshots,
        'agent_positions': agent_positions
    }
    
    return simulation_results
