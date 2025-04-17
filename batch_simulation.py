import numpy as np
import pandas as pd
import itertools
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json

# Import our simulation module
import simulation
import database

class BatchSimulator:
    """
    Handles the execution of multiple simulations with varying parameters
    for systematic experimentation and statistical analysis.
    """
    
    def __init__(self):
        self.results = []
        self.parameter_space = {}
        self.metadata = {
            "created_at": datetime.now(),
            "total_simulations": 0,
            "completed_simulations": 0,
            "success_rate": 0,
            "avg_runtime": 0,
            "hypothesis": None
        }
    
    def set_parameter_space(self, parameter_dict):
        """
        Define the parameter space to explore.
        Each parameter can be a single value or a list of values to try.
        
        Example:
        {
            'steps': [50, 100, 200],
            'grid_size': [10, 15, 20],
            'resources': [5, 10, 15],
            'dangers': [3, 5, 8],
            'goals': [1, 2]
        }
        """
        self.parameter_space = parameter_dict
        
        # Calculate total number of simulations
        param_combinations = 1
        for param, values in parameter_dict.items():
            if isinstance(values, list):
                param_combinations *= len(values)
            else:
                # Single value
                param_combinations *= 1
        
        self.metadata["total_simulations"] = param_combinations
        return param_combinations
    
    def set_hypothesis(self, hypothesis_text, test_type="correlation", 
                       variables=None, expected_outcome=None, threshold=0.05):
        """
        Define a hypothesis to test with this batch of simulations.
        
        Args:
            hypothesis_text: Plain text description of the hypothesis
            test_type: Type of statistical test (correlation, t-test, etc.)
            variables: Variables involved in the hypothesis
            expected_outcome: Expected result
            threshold: Significance threshold (p-value)
        """
        self.metadata["hypothesis"] = {
            "text": hypothesis_text,
            "test_type": test_type,
            "variables": variables if variables else [],
            "expected_outcome": expected_outcome,
            "threshold": threshold,
            "result": None,
            "confirmed": None
        }
    
    def _generate_parameter_combinations(self):
        """Generate all combinations of parameters to test"""
        # Convert single values to lists for uniform processing
        param_space = {}
        for param, values in self.parameter_space.items():
            if not isinstance(values, list):
                param_space[param] = [values]
            else:
                param_space[param] = values
        
        # Get parameter names and values
        param_names = list(param_space.keys())
        param_values = [param_space[param] for param in param_names]
        
        # Generate all combinations
        combinations = list(itertools.product(*param_values))
        
        # Convert to list of dictionaries
        param_combinations = []
        for combo in combinations:
            param_dict = {param_names[i]: combo[i] for i in range(len(param_names))}
            param_combinations.append(param_dict)
        
        return param_combinations

    def run_batch(self, max_workers=None, save_to_db=True, experiment_name=None):
        """
        Run all simulations in the parameter space.
        
        Args:
            max_workers: Max number of parallel simulations (None = auto)
            save_to_db: Whether to save individual simulations to the database
            experiment_name: Name for this batch experiment
        
        Returns:
            DataFrame with aggregated results
        """
        param_combinations = self._generate_parameter_combinations()
        self.results = []
        
        start_time = time.time()
        completed = 0
        success = 0
        
        # Run simulations (potentially in parallel)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for params in param_combinations:
                futures.append(executor.submit(self._run_single_simulation, params))
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        self.results.append(result)
                        success += 1
                        
                        # Save to database if requested
                        if save_to_db and experiment_name:
                            try:
                                sim_id = database.save_simulation(
                                    result["simulation_data"], 
                                    name=f"{experiment_name} - Run {completed+1}",
                                    description=f"Part of batch experiment: {experiment_name}\nParameters: {json.dumps(result['parameters'])}"
                                )
                                result["database_id"] = sim_id
                            except Exception as e:
                                print(f"Error saving simulation to database: {str(e)}")
                    
                    completed += 1
                    
                    # Update metadata
                    self.metadata["completed_simulations"] = completed
                    self.metadata["success_rate"] = (success / max(1, completed)) * 100
                    
                except Exception as e:
                    print(f"Error in simulation: {str(e)}")
        
        # Calculate runtime statistics
        total_runtime = time.time() - start_time
        self.metadata["avg_runtime"] = total_runtime / max(1, len(self.results))
        self.metadata["total_runtime"] = total_runtime
        
        # Process and aggregate results
        return self.aggregate_results()
    
    def _run_single_simulation(self, params):
        """Run a single simulation with the given parameters"""
        try:
            # Extract simulation parameters
            steps = params.get('steps', 100)
            grid_size = params.get('grid_size', 10)
            resources = params.get('resources', 10)
            dangers = params.get('dangers', 5)
            goals = params.get('goals', 1)
            
            # Handle agent starting positions
            start_pos_a = params.get('start_pos_a', (None, None))
            start_pos_b = params.get('start_pos_b', (None, None))
            
            # Run the simulation
            sim_results = simulation.run_simulation(
                steps=steps,
                grid_size=grid_size,
                resources=resources,
                dangers=dangers,
                goals=goals,
                start_pos_a=start_pos_a,
                start_pos_b=start_pos_b
            )
            
            # Extract key metrics
            log_df = sim_results['log']
            
            # Calculate statistics
            stats = {
                "parameters": params,
                "avg_resonance": float(log_df['resonance'].mean()),
                "max_resonance": float(log_df['resonance'].max()),
                "min_resonance": float(log_df['resonance'].min()),
                "avg_bond_strength_a": float(log_df['A_bond'].mean()),
                "avg_bond_strength_b": float(log_df['B_bond'].mean()),
                "final_relationship_score": float(sim_results['relationship_score']),
                "A_phi_mean": float(log_df['A_phi'].mean()),
                "A_sigma_mean": float(log_df['A_sigma'].mean()),
                "A_delta_mean": float(log_df['A_delta'].mean()),
                "B_phi_mean": float(log_df['B_phi'].mean()),
                "B_sigma_mean": float(log_df['B_sigma'].mean()),
                "B_delta_mean": float(log_df['B_delta'].mean()),
                "distance_traveled_A": int(_calculate_distance(log_df, 'A')),
                "distance_traveled_B": int(_calculate_distance(log_df, 'B')),
                "resource_encounters": int(_count_events(log_df, ['A_found_resource', 'B_found_resource'])),
                "danger_encounters": int(_count_events(log_df, ['A_found_danger', 'B_found_danger'])),
                "goal_encounters": int(_count_events(log_df, ['A_found_goal', 'B_found_goal'])),
                "time_together": int(_count_events(log_df, ['A_sees_B'], True)),
                "simulation_data": sim_results  # Store full simulation data
            }
            
            return stats
            
        except Exception as e:
            print(f"Error running simulation: {str(e)}")
            return None
    
    def aggregate_results(self):
        """Aggregate results into a pandas DataFrame for analysis"""
        if not self.results:
            return pd.DataFrame()
        
        # Flatten the results for DataFrame creation
        flat_data = []
        for result in self.results:
            # Skip simulation_data to avoid huge DataFrame
            data = {k: v for k, v in result.items() if k != 'simulation_data'}
            
            # Flatten parameters
            for param_name, param_value in data['parameters'].items():
                data[f"param_{param_name}"] = param_value
            
            # Remove the parameters dict to avoid duplication
            del data['parameters']
            
            flat_data.append(data)
        
        return pd.DataFrame(flat_data)
    
    def test_hypothesis(self):
        """Test the hypothesis defined for this batch of simulations"""
        if not self.metadata["hypothesis"] or not self.results:
            return None
        
        hypothesis = self.metadata["hypothesis"]
        df = self.aggregate_results()
        
        # Run appropriate statistical test
        if hypothesis["test_type"] == "correlation":
            if len(hypothesis["variables"]) >= 2:
                var1, var2 = hypothesis["variables"][:2]
                if var1 in df.columns and var2 in df.columns:
                    # Calculate correlation
                    correlation = df[var1].corr(df[var2])
                    p_value = 0.05  # Placeholder - would need scipy for real p-value
                    
                    # Update hypothesis result
                    hypothesis["result"] = {
                        "correlation": correlation,
                        "p_value": p_value,
                        "n": len(df)
                    }
                    
                    # Check if hypothesis is confirmed
                    if hypothesis["expected_outcome"] == "positive":
                        hypothesis["confirmed"] = correlation > 0 and p_value < hypothesis["threshold"]
                    elif hypothesis["expected_outcome"] == "negative":
                        hypothesis["confirmed"] = correlation < 0 and p_value < hypothesis["threshold"]
                    else:
                        hypothesis["confirmed"] = p_value < hypothesis["threshold"]
        
        # Additional test types would go here
        
        return hypothesis

# Helper functions
def _calculate_distance(df, agent_prefix):
    """Calculate the total distance traveled by an agent"""
    if len(df) <= 1:
        return 0
    
    # Calculate Manhattan distance between consecutive positions
    x_col, y_col = f"{agent_prefix}_x", f"{agent_prefix}_y"
    
    if x_col not in df.columns or y_col not in df.columns:
        return 0
    
    x_diff = df[x_col].diff().abs().fillna(0)
    y_diff = df[y_col].diff().abs().fillna(0)
    
    return (x_diff + y_diff).sum()

def _count_events(df, columns, value=True):
    """Count the number of events in the dataframe"""
    count = 0
    for col in columns:
        if col in df.columns:
            count += df[col].eq(value).sum()
    return count