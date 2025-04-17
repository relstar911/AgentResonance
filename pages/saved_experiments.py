import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime
import matplotlib.pyplot as plt
import uuid
from database import get_all_experiments, get_experiment, delete_experiment

# Page title
st.title("Saved Experiments")
st.write("View and analyze your saved experiments.")

# Get all experiments from database
try:
    experiments = get_all_experiments()
    
    if not experiments:
        st.info("No saved experiments found. Go to the Experiment Lab to create and save experiments.")
    else:
        # Create a simple table of experiments
        exp_data = []
        for exp in experiments:
            # Extract independent variable info if possible
            try:
                ind_var = json.loads(exp.independent_variable) if exp.independent_variable else {}
                ind_var_name = ind_var.get("display_name", "N/A")
            except:
                ind_var_name = "N/A"
            
            # Extract dependent variables if possible
            try:
                dep_vars = json.loads(exp.dependent_variables) if exp.dependent_variables else []
                dep_vars_count = len(dep_vars)
            except:
                dep_vars_count = 0
            
            exp_data.append({
                "ID": exp.id,
                "Name": exp.name,
                "Created": exp.created_at.strftime("%Y-%m-%d %H:%M"),
                "Independent Variable": ind_var_name,
                "Dependent Variables": dep_vars_count,
                "Replications": exp.replications
            })
        
        # Convert to DataFrame
        exp_df = pd.DataFrame(exp_data)
        
        # Let the user select an experiment
        st.subheader("Select an Experiment")
        
        # Use a dataframe as a selector
        st.dataframe(exp_df, use_container_width=True)
        
        selected_exp_id = st.selectbox("Select Experiment ID", 
                                        options=[exp.id for exp in experiments],
                                        format_func=lambda x: f"ID: {x} - {next((e.name for e in experiments if e.id == x), 'Unknown')}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Load Experiment", type="primary"):
                # Get full experiment data
                experiment = get_experiment(selected_exp_id)
                
                if experiment:
                    # Store in session state for experiment lab
                    st.session_state.experiment_stage = "analysis"
                    st.session_state.experiment_design = {
                        "name": experiment["name"],
                        "hypothesis": experiment["hypothesis"],
                        "independent_variable": experiment["independent_variable"],
                        "dependent_variables": experiment["dependent_variables"],
                        "control_group": experiment["control_group"],
                        "experimental_groups": experiment["experimental_groups"],
                        "replications": experiment["replications"],
                        "randomization": experiment["randomization"]
                    }
                    
                    st.session_state.experiment_results = {
                        "control_data": experiment["control_results"],
                        "experimental_data": experiment["experimental_results"],
                        "analysis": experiment["analysis"]
                    }
                    
                    st.success(f"Experiment '{experiment['name']}' loaded successfully!")
                    st.info("Go to the Experiment Lab page to view the analysis.")
                else:
                    st.error("Failed to load experiment. The experiment may no longer exist.")
        
        with col2:
            if st.button("Delete Experiment"):
                if st.session_state.get("confirm_delete") == selected_exp_id:
                    # User already confirmed, proceed with deletion
                    if delete_experiment(selected_exp_id):
                        st.success("Experiment deleted successfully!")
                        # Clear confirmation state
                        st.session_state.pop("confirm_delete", None)
                        # Refresh the page
                        st.rerun()
                    else:
                        st.error("Failed to delete experiment.")
                else:
                    # Ask for confirmation
                    st.session_state.confirm_delete = selected_exp_id
                    st.warning(f"Are you sure you want to delete experiment ID: {selected_exp_id}? Click 'Delete Experiment' again to confirm.")

        # Show experiment details
        if st.session_state.get("confirm_delete") == selected_exp_id:
            st.warning("⚠️ This experiment will be deleted if you confirm the operation above.")
        
        st.subheader("Experiment Details")
        
        # Get selected experiment
        selected_exp = next((e for e in experiments if e.id == selected_exp_id), None)
        
        if selected_exp:
            # Display basic info
            st.write(f"**Name:** {selected_exp.name}")
            st.write(f"**Created:** {selected_exp.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Try to display hypothesis
            if selected_exp.hypothesis:
                st.write(f"**Hypothesis:** {selected_exp.hypothesis}")
            
            # Try to display independent variable
            try:
                ind_var = json.loads(selected_exp.independent_variable) if selected_exp.independent_variable else {}
                if ind_var and "display_name" in ind_var:
                    st.write(f"**Independent Variable:** {ind_var['display_name']}")
                    st.write(f"**Control Value:** {ind_var.get('control_value', 'N/A')} {ind_var.get('unit', '')}")
                    st.write(f"**Experimental Values:** {', '.join(map(str, ind_var.get('experimental_values', [])))}")
            except:
                st.write("**Independent Variable:** Could not parse data")
            
            # Try to display dependent variables
            try:
                dep_vars = json.loads(selected_exp.dependent_variables) if selected_exp.dependent_variables else []
                if dep_vars:
                    var_names = {
                        "relationship_score": "Relationship Score",
                        "avg_resonance": "Average Resonance",
                        "max_resonance": "Maximum Resonance",
                        "avg_bond_strength_a": "Agent A Avg Bond Strength",
                        "avg_bond_strength_b": "Agent B Avg Bond Strength",
                        "A_phi_mean": "Agent A Phi (Integration)",
                        "A_sigma_mean": "Agent A Sigma (Self-predictive Accuracy)",
                        "A_delta_mean": "Agent A Delta (Decision Entropy)",
                        "B_phi_mean": "Agent B Phi (Integration)",
                        "B_sigma_mean": "Agent B Sigma (Self-predictive Accuracy)",
                        "B_delta_mean": "Agent B Delta (Decision Entropy)",
                        "time_together": "Time Spent Together"
                    }
                    
                    var_display_names = [var_names.get(var, var) for var in dep_vars]
                    st.write(f"**Dependent Variables:** {', '.join(var_display_names)}")
            except:
                st.write("**Dependent Variables:** Could not parse data")
            
            st.write(f"**Replications:** {selected_exp.replications}")
            st.write(f"**Randomization:** {'Enabled' if selected_exp.randomization else 'Disabled'}")
        
except Exception as e:
    st.error(f"Error loading experiments: {str(e)}")
    st.info("Please ensure that the database is properly configured and contains experiment data.")