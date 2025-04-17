import os
import json
import pandas as pd
import numpy as np
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Create database connection
Base = declarative_base()

# Database URL from environment variable
database_url = os.environ.get('DATABASE_URL')
if not database_url:
    raise ValueError("DATABASE_URL environment variable not set")

# Create engine and session with connection pooling and better error handling
engine = create_engine(
    database_url,
    pool_pre_ping=True,  # Check connection before using
    pool_recycle=3600,   # Recycle connections after 1 hour
    pool_timeout=30,     # Wait 30 seconds for a connection
    connect_args={"connect_timeout": 10}  # 10 seconds connection timeout
)
Session = sessionmaker(bind=engine)

class SimulationRun(Base):
    """Table to store metadata about each simulation run"""
    __tablename__ = 'simulation_runs'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    grid_size = Column(Integer)
    resources = Column(Integer)
    dangers = Column(Integer)
    goals = Column(Integer)
    steps = Column(Integer)
    agent_a_start_x = Column(Integer, nullable=True)
    agent_a_start_y = Column(Integer, nullable=True)
    agent_b_start_x = Column(Integer, nullable=True)
    agent_b_start_y = Column(Integer, nullable=True)
    relationship_score = Column(Float)
    avg_resonance = Column(Float)
    avg_bond_strength_a = Column(Float)
    avg_bond_strength_b = Column(Float)
    agent_positions = Column(Text)  # Stored as JSON
    grid_data = Column(Text)  # Stored as JSON
    full_log = Column(Text)  # Stored as JSON


class ExperimentRun(Base):
    """Table to store metadata and results about each experiment"""
    __tablename__ = 'experiment_runs'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    hypothesis = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    independent_variable = Column(Text)  # Stored as JSON
    dependent_variables = Column(Text)  # Stored as JSON list
    control_group = Column(Text)  # Stored as JSON
    experimental_groups = Column(Text)  # Stored as JSON
    replications = Column(Integer)
    randomization = Column(Integer, default=1)  # Using Integer instead of Boolean for SQLite compatibility
    control_results = Column(Text)  # Stored as JSON
    experimental_results = Column(Text)  # Stored as JSON
    analysis = Column(Text)  # Stored as JSON

def init_db():
    """Initialize the database by creating tables"""
    Base.metadata.create_all(engine)
    print("Database tables created.")

def save_simulation(simulation_results, name=None, description=None):
    """Save simulation results to the database"""
    # Extract data from simulation results
    grid = simulation_results['grid'].tolist()
    log_df = simulation_results['log']
    agent_a = simulation_results['agent_a']
    agent_b = simulation_results['agent_b']
    
    # Convert NumPy types to Python native types
    relationship_score = float(simulation_results['relationship_score'])
    
    # Convert agent positions to ensure they're serializable
    agent_positions = []
    for pos in simulation_results['agent_positions']:
        agent_positions.append([list(map(int, pos[0])), list(map(int, pos[1]))])
    
    # Calculate averages and convert to native Python types
    avg_resonance = float(log_df['resonance'].mean())
    avg_bond_strength_a = float(log_df['A_bond'].mean())
    avg_bond_strength_b = float(log_df['B_bond'].mean())
    
    # Convert all DataFrame columns to Python native types to prevent NumPy type errors
    for column in log_df.columns:
        if log_df[column].dtype.name.startswith(('float', 'int')):
            log_df[column] = log_df[column].astype(float)
    
    # Create simulation record
    simulation = SimulationRun(
        name=name,
        description=description,
        grid_size=len(grid),
        resources=int(simulation_results.get('resources', 0)),
        dangers=int(simulation_results.get('dangers', 0)),
        goals=int(simulation_results.get('goals', 0)),
        steps=len(log_df),
        agent_a_start_x=int(agent_a.x),  # Convert to int
        agent_a_start_y=int(agent_a.y),  # Convert to int
        agent_b_start_x=int(agent_b.x),  # Convert to int
        agent_b_start_y=int(agent_b.y),  # Convert to int
        relationship_score=relationship_score,
        avg_resonance=avg_resonance,
        avg_bond_strength_a=avg_bond_strength_a,
        avg_bond_strength_b=avg_bond_strength_b,
        agent_positions=json.dumps(agent_positions),
        grid_data=json.dumps(grid),
        full_log=log_df.to_json(orient='records')
    )
    
    # Save to database
    session = Session()
    try:
        session.add(simulation)
        session.commit()
        sim_id = simulation.id
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()
    
    return sim_id

def get_all_simulations():
    """Get metadata for all simulations"""
    session = Session()
    try:
        simulations = session.query(SimulationRun).order_by(SimulationRun.created_at.desc()).all()
        return simulations
    except Exception as e:
        print(f"Database error in get_all_simulations: {str(e)}")
        return []  # Return empty list on error
    finally:
        session.close()

def get_simulation(simulation_id):
    """Get full data for a single simulation"""
    session = Session()
    try:
        simulation = session.query(SimulationRun).filter(SimulationRun.id == simulation_id).first()
        if not simulation:
            return None
        
        # Create a dictionary with all the simulation data
        try:
            # Make sure all values are Python native types
            sim_data = {
                'id': int(simulation.id),
                'name': str(simulation.name) if simulation.name else None,
                'description': str(simulation.description) if simulation.description else None,
                'created_at': simulation.created_at,
                'grid_size': int(simulation.grid_size),
                'resources': int(simulation.resources),
                'dangers': int(simulation.dangers),
                'goals': int(simulation.goals),
                'steps': int(simulation.steps),
                'agent_a_start_x': int(simulation.agent_a_start_x) if simulation.agent_a_start_x is not None else None,
                'agent_a_start_y': int(simulation.agent_a_start_y) if simulation.agent_a_start_y is not None else None,
                'agent_b_start_x': int(simulation.agent_b_start_x) if simulation.agent_b_start_x is not None else None,
                'agent_b_start_y': int(simulation.agent_b_start_y) if simulation.agent_b_start_y is not None else None,
                'relationship_score': float(simulation.relationship_score),
                'avg_resonance': float(simulation.avg_resonance),
                'avg_bond_strength_a': float(simulation.avg_bond_strength_a),
                'avg_bond_strength_b': float(simulation.avg_bond_strength_b),
                'grid': json.loads(simulation.grid_data),
                'agent_positions': json.loads(simulation.agent_positions),
                'log': pd.read_json(simulation.full_log, orient='records')
            }
            return sim_data
        except Exception as e:
            print(f"Error parsing simulation data: {str(e)}")
            return None
    except Exception as e:
        print(f"Database error in get_simulation: {str(e)}")
        return None
    finally:
        session.close()

def delete_simulation(simulation_id):
    """Delete a simulation from the database"""
    session = Session()
    try:
        simulation = session.query(SimulationRun).filter(SimulationRun.id == simulation_id).first()
        if simulation:
            session.delete(simulation)
            session.commit()
            return True
        return False
    except:
        session.rollback()
        return False
    finally:
        session.close()


def save_experiment(experiment_data):
    """Save experiment design and results to the database"""
    session = Session()
    try:
        # Deep copy to avoid modifying original data
        data_copy = experiment_data.copy()
        
        # Helper function to convert numpy arrays and other non-serializable types
        def convert_numpy_types(obj):
            if isinstance(obj, pd.DataFrame):
                # Convert DataFrame to dict of lists
                result = {}
                for column in obj.columns:
                    if obj[column].dtype.name.startswith(('float', 'int')):
                        result[column] = obj[column].astype(float).tolist()
                    elif np.issubdtype(obj[column].dtype, np.ndarray):
                        result[column] = obj[column].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x).tolist()
                    else:
                        result[column] = obj[column].tolist()
                return result
            elif isinstance(obj, pd.Series):
                # Convert Series to list
                if obj.dtype.name.startswith(('float', 'int')):
                    return obj.astype(float).tolist()
                else:
                    return obj.tolist()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
                return int(obj)
            elif hasattr(np, 'float_') and isinstance(obj, np.float_):  # Für ältere NumPy-Versionen
                return float(obj)
            elif hasattr(np, 'int_') and isinstance(obj, np.int_):  # Für ältere NumPy-Versionen
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy_types(item) for item in obj)
            else:
                return obj
        
        # Convert the entire experiment data structure
        serializable_data = convert_numpy_types(data_copy)
        
        # Extract specific components
        independent_variable = json.dumps(serializable_data.get("independent_variable", {}))
        dependent_variables = json.dumps(serializable_data.get("dependent_variables", []))
        control_group = json.dumps(serializable_data.get("control_group", {}))
        experimental_groups = json.dumps(serializable_data.get("experimental_groups", []))
        
        # Convert experimental_results to JSON strings
        if "experimental_results" in serializable_data and serializable_data["experimental_results"]:
            exp_results = []
            for group_results in serializable_data["experimental_results"]:
                # Each result is already converted to a serializable form
                exp_results.append(json.dumps(group_results))
        else:
            exp_results = []
        
        # Convert control_results to JSON string
        if "control_results" in serializable_data and serializable_data["control_results"] is not None:
            # Already converted to a serializable form
            control_results = json.dumps(serializable_data["control_results"])
        else:
            control_results = None
        
        # Convert analysis results to JSON string
        if "analysis" in serializable_data and serializable_data["analysis"]:
            # Already converted to a serializable form
            analysis = json.dumps(serializable_data["analysis"])
        else:
            analysis = None
        
        # Create experiment record
        experiment = ExperimentRun(
            name=experiment_data.get("name", "Unnamed Experiment"),
            hypothesis=experiment_data.get("hypothesis"),
            independent_variable=json.dumps(experiment_data.get("independent_variable", {})),
            dependent_variables=json.dumps(experiment_data.get("dependent_variables", [])),
            control_group=json.dumps(experiment_data.get("control_group", {})),
            experimental_groups=json.dumps(experiment_data.get("experimental_groups", [])),
            replications=experiment_data.get("replications", 1),
            randomization=1 if experiment_data.get("randomization", True) else 0,
            control_results=control_results,
            experimental_results=json.dumps(exp_results),
            analysis=analysis
        )
        
        session.add(experiment)
        session.commit()
        exp_id = experiment.id
        return exp_id
    
    except Exception as e:
        session.rollback()
        print(f"Error saving experiment: {str(e)}")
        raise e
    finally:
        session.close()


def get_all_experiments():
    """Get metadata for all experiments"""
    session = Session()
    try:
        experiments = session.query(ExperimentRun).order_by(ExperimentRun.created_at.desc()).all()
        return experiments
    except Exception as e:
        print(f"Database error in get_all_experiments: {str(e)}")
        return []  # Return empty list on error
    finally:
        session.close()


def get_experiment(experiment_id):
    """Get full data for a single experiment"""
    session = Session()
    try:
        experiment = session.query(ExperimentRun).filter(ExperimentRun.id == experiment_id).first()
        if not experiment:
            return None
        
        # Create a dictionary with all the experiment data
        try:
            # Convert JSON strings back to Python objects
            exp_data = {
                'id': int(experiment.id),
                'name': str(experiment.name),
                'hypothesis': str(experiment.hypothesis) if experiment.hypothesis else None,
                'created_at': experiment.created_at,
                'replications': int(experiment.replications),
                'randomization': bool(experiment.randomization),
                'independent_variable': json.loads(experiment.independent_variable) if experiment.independent_variable else {},
                'dependent_variables': json.loads(experiment.dependent_variables) if experiment.dependent_variables else [],
                'control_group': json.loads(experiment.control_group) if experiment.control_group else {},
                'experimental_groups': json.loads(experiment.experimental_groups) if experiment.experimental_groups else []
            }
            
            # Handle results data (convert JSON to DataFrames where needed)
            if experiment.control_results:
                try:
                    exp_data['control_results'] = pd.read_json(experiment.control_results, orient='records')
                except:
                    exp_data['control_results'] = json.loads(experiment.control_results)
            else:
                exp_data['control_results'] = None
            
            if experiment.experimental_results:
                exp_results_json = json.loads(experiment.experimental_results)
                exp_results = []
                for group_result in exp_results_json:
                    try:
                        exp_results.append(pd.read_json(group_result, orient='records'))
                    except:
                        exp_results.append(json.loads(group_result))
                exp_data['experimental_results'] = exp_results
            else:
                exp_data['experimental_results'] = []
            
            if experiment.analysis:
                exp_data['analysis'] = json.loads(experiment.analysis)
            else:
                exp_data['analysis'] = {}
            
            return exp_data
        
        except Exception as e:
            print(f"Error parsing experiment data: {str(e)}")
            return None
    
    except Exception as e:
        print(f"Database error in get_experiment: {str(e)}")
        return None
    
    finally:
        session.close()


def delete_experiment(experiment_id):
    """Delete an experiment from the database"""
    session = Session()
    try:
        experiment = session.query(ExperimentRun).filter(ExperimentRun.id == experiment_id).first()
        if experiment:
            session.delete(experiment)
            session.commit()
            return True
        return False
    except Exception as e:
        session.rollback()
        print(f"Error deleting experiment: {str(e)}")
        return False
    finally:
        session.close()


# Initialize database tables when importing this module
init_db()