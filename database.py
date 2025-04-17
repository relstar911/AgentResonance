import os
import json
import pandas as pd
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
    relationship_score = simulation_results['relationship_score']
    agent_positions = simulation_results['agent_positions']
    
    # Calculate averages
    avg_resonance = log_df['resonance'].mean()
    avg_bond_strength_a = log_df['A_bond'].mean()
    avg_bond_strength_b = log_df['B_bond'].mean()
    
    # Create simulation record
    simulation = SimulationRun(
        name=name,
        description=description,
        grid_size=len(grid),
        resources=simulation_results.get('resources', 0),
        dangers=simulation_results.get('dangers', 0),
        goals=simulation_results.get('goals', 0),
        steps=len(log_df),
        agent_a_start_x=agent_a.x,  # Corrected from start_x to x
        agent_a_start_y=agent_a.y,  # Corrected from start_y to y
        agent_b_start_x=agent_b.x,  # Corrected from start_x to x
        agent_b_start_y=agent_b.y,  # Corrected from start_y to y
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
            sim_data = {
                'id': simulation.id,
                'name': simulation.name,
                'description': simulation.description,
                'created_at': simulation.created_at,
                'grid_size': simulation.grid_size,
                'resources': simulation.resources,
                'dangers': simulation.dangers,
                'goals': simulation.goals,
                'steps': simulation.steps,
                'agent_a_start_x': simulation.agent_a_start_x,
                'agent_a_start_y': simulation.agent_a_start_y,
                'agent_b_start_x': simulation.agent_b_start_x,
                'agent_b_start_y': simulation.agent_b_start_y,
                'relationship_score': simulation.relationship_score,
                'avg_resonance': simulation.avg_resonance,
                'avg_bond_strength_a': simulation.avg_bond_strength_a,
                'avg_bond_strength_b': simulation.avg_bond_strength_b,
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

# Initialize database tables when importing this module
init_db()