import os
import json
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Connect to the PostgreSQL database
DATABASE_URL = os.environ.get('DATABASE_URL')
engine = create_engine(DATABASE_URL)
Base = declarative_base()
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

def save_simulation(simulation_results, name=None, description=None):
    """Save simulation results to the database"""
    # Extract data from simulation_results
    grid = simulation_results['grid']
    log_df = simulation_results['log']
    agent_a = simulation_results['agent_a']
    agent_b = simulation_results['agent_b']
    relationship_score = simulation_results['relationship_score']
    agent_positions = simulation_results['agent_positions']
    
    # Calculate average metrics
    avg_resonance = log_df['resonance'].mean()
    avg_bond_a = log_df['A_bond'].mean()
    avg_bond_b = log_df['B_bond'].mean()
    
    # Create new simulation record
    sim = SimulationRun(
        name=name,
        description=description,
        created_at=datetime.now(),
        grid_size=len(grid),
        resources=len(grid[grid == 1]),
        dangers=len(grid[grid == -1]),
        goals=len(grid[grid == 2]),
        steps=len(log_df),
        agent_a_start_x=agent_a.path_history[0][0] if agent_a.path_history else None,
        agent_a_start_y=agent_a.path_history[0][1] if agent_a.path_history else None,
        agent_b_start_x=agent_b.path_history[0][0] if agent_b.path_history else None,
        agent_b_start_y=agent_b.path_history[0][1] if agent_b.path_history else None,
        relationship_score=relationship_score,
        avg_resonance=avg_resonance,
        avg_bond_strength_a=avg_bond_a,
        avg_bond_strength_b=avg_bond_b,
        agent_positions=json.dumps([[(pos[0][0], pos[0][1]), (pos[1][0], pos[1][1])] for pos in agent_positions]),
        grid_data=json.dumps(grid.tolist()),
        full_log=log_df.to_json(orient='records')
    )
    
    # Save to database
    session = Session()
    try:
        session.add(sim)
        session.commit()
        return sim.id
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_all_simulations():
    """Get metadata for all simulations"""
    session = Session()
    try:
        sims = session.query(SimulationRun).order_by(SimulationRun.created_at.desc()).all()
        return sims
    finally:
        session.close()

def get_simulation(simulation_id):
    """Get full data for a single simulation"""
    session = Session()
    try:
        sim = session.query(SimulationRun).filter(SimulationRun.id == simulation_id).first()
        if not sim:
            return None
        
        # Convert stored JSON back to appropriate data structures
        result = {
            'id': sim.id,
            'name': sim.name,
            'description': sim.description,
            'created_at': sim.created_at,
            'grid_size': sim.grid_size,
            'resources': sim.resources,
            'dangers': sim.dangers,
            'goals': sim.goals,
            'steps': sim.steps,
            'agent_a_start': (sim.agent_a_start_x, sim.agent_a_start_y),
            'agent_b_start': (sim.agent_b_start_x, sim.agent_b_start_y),
            'relationship_score': sim.relationship_score,
            'avg_resonance': sim.avg_resonance,
            'avg_bond_strength_a': sim.avg_bond_strength_a,
            'avg_bond_strength_b': sim.avg_bond_strength_b,
            'agent_positions': json.loads(sim.agent_positions) if sim.agent_positions else [],
            'grid': json.loads(sim.grid_data) if sim.grid_data else [],
            'log': pd.read_json(sim.full_log) if sim.full_log else pd.DataFrame()
        }
        return result
    finally:
        session.close()

def delete_simulation(simulation_id):
    """Delete a simulation from the database"""
    session = Session()
    try:
        sim = session.query(SimulationRun).filter(SimulationRun.id == simulation_id).first()
        if sim:
            session.delete(sim)
            session.commit()
            return True
        return False
    finally:
        session.close()

# Initialize the database
init_db()