import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import json
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import itertools
import uuid
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
import simulation
import visualization
import analysis
import database
import batch_simulation

# Page configuration
st.set_page_config(
    page_title="Experiment Laboratory",
    page_icon="🔬",
    layout="wide"
)

# Title and description
st.title("Experiment Laboratory")
st.markdown("""
Design and conduct structured experiments to test specific hypotheses about conscious agent behavior.
This laboratory provides tools for proper experimental design with control and experimental groups.
""")

# Initialize session state for experiment setup
if "experiment_stage" not in st.session_state:
    st.session_state.experiment_stage = "design"  # design, running, analysis
if "experiment_design" not in st.session_state:
    st.session_state.experiment_design = {
        "experiment_id": str(uuid.uuid4()),
        "name": f"Experiment {datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "description": "",
        "hypothesis": "",
        "independent_variable": {},
        "control_group": {},
        "experimental_groups": [],
        "dependent_variables": [],
        "replications": 5,
        "randomization": True
    }
if "experiment_results" not in st.session_state:
    st.session_state.experiment_results = {
        "control_data": None,
        "experimental_data": [],
        "summary": None,
        "statistical_tests": None
    }

# Functions for experiments
def run_experiment_group(params, replications, group_name, progress_callback=None):
    """Run a group of simulations with the same parameters"""
    results = []
    
    for i in range(replications):
        if progress_callback:
            progress_callback(f"Running {group_name} - replication {i+1}/{replications}")
        
        # Run a single simulation
        try:
            sim_result = simulation.run_simulation(**params)
            
            # Extract key metrics
            log_df = sim_result['log']
            
            # Calculate statistics
            stats = {
                "replication": i+1,
                "group": group_name,
                "parameters": params,
                "relationship_score": float(sim_result['relationship_score']),
                "avg_resonance": float(log_df['resonance'].mean()),
                "max_resonance": float(log_df['resonance'].max()),
                "min_resonance": float(log_df['resonance'].min()),
                "avg_bond_strength_a": float(log_df['A_bond'].mean()),
                "avg_bond_strength_b": float(log_df['B_bond'].mean()),
                "final_relationship_score": float(sim_result['relationship_score']),
                "A_phi_mean": float(log_df['A_phi'].mean()),
                "A_sigma_mean": float(log_df['A_sigma'].mean()),
                "A_delta_mean": float(log_df['A_delta'].mean()),
                "B_phi_mean": float(log_df['B_phi'].mean()),
                "B_sigma_mean": float(log_df['B_sigma'].mean()),
                "B_delta_mean": float(log_df['B_delta'].mean()),
                "time_together": int(log_df['A_sees_B'].sum()),
                "simulation_data": sim_result  # Store full simulation data
            }
            
            results.append(stats)
            
        except Exception as e:
            st.error(f"Error running simulation: {str(e)}")
    
    return results

def analyze_experiment_results(control_data, experimental_data, dependent_variables, independent_variable):
    """Analyze experimental results and perform statistical tests"""
    
    # Convert to dataframes
    control_df = pd.DataFrame([{k: v for k, v in result.items() if k != 'simulation_data'} for result in control_data])
    
    # Combine all experimental groups
    exp_dfs = []
    for group_idx, group_data in enumerate(experimental_data):
        if group_data:
            group_df = pd.DataFrame([{k: v for k, v in result.items() if k != 'simulation_data'} for result in group_data])
            exp_dfs.append(group_df)
    
    if not exp_dfs:
        return {"error": "No experimental data available for analysis"}
    
    exp_df = pd.concat(exp_dfs, ignore_index=True)
    combined_df = pd.concat([control_df, exp_df], ignore_index=True)
    
    # Statistical tests for each dependent variable
    test_results = {}
    
    for var in dependent_variables:
        # T-tests between control and each experimental group
        group_results = []
        
        for i, group_df in enumerate(exp_dfs):
            if len(control_df) > 0 and len(group_df) > 0:
                try:
                    t_stat, p_val = stats.ttest_ind(
                        control_df[var].values,
                        group_df[var].values,
                        equal_var=False  # Welch's t-test (doesn't assume equal variance)
                    )
                    
                    # Calculate effect size (Cohen's d)
                    mean1 = control_df[var].mean()
                    mean2 = group_df[var].mean()
                    std1 = control_df[var].std()
                    std2 = group_df[var].std()
                    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
                    effect_size = abs(mean1 - mean2) / pooled_std if pooled_std > 0 else 0
                    
                    group_results.append({
                        "group_name": f"Experimental Group {i+1}",
                        "t_statistic": t_stat,
                        "p_value": p_val,
                        "significant": p_val < 0.05,
                        "control_mean": mean1,
                        "experimental_mean": mean2,
                        "mean_difference": mean2 - mean1,
                        "mean_difference_percent": ((mean2 - mean1) / mean1 * 100) if mean1 != 0 else 0,
                        "effect_size": effect_size,
                        "effect_size_interpretation": ("small" if effect_size < 0.5 else 
                                                     "medium" if effect_size < 0.8 else "large")
                    })
                except Exception as e:
                    group_results.append({
                        "group_name": f"Experimental Group {i+1}",
                        "error": str(e)
                    })
        
        test_results[var] = group_results
    
    # ANOVA for comparing all groups
    anova_results = {}
    for var in dependent_variables:
        try:
            groups = [control_df[var].values]
            for group_df in exp_dfs:
                groups.append(group_df[var].values)
            
            if all(len(g) > 0 for g in groups) and len(groups) >= 2:
                f_val, p_val = stats.f_oneway(*groups)
                anova_results[var] = {
                    "f_statistic": f_val,
                    "p_value": p_val,
                    "significant": p_val < 0.05
                }
            else:
                anova_results[var] = {"error": "Insufficient data for ANOVA"}
        except Exception as e:
            anova_results[var] = {"error": str(e)}
    
    # Summary statistics
    summary = {}
    for var in dependent_variables:
        group_stats = [
            {
                "group": "Control Group",
                "mean": control_df[var].mean() if not control_df.empty else None,
                "std": control_df[var].std() if not control_df.empty else None,
                "min": control_df[var].min() if not control_df.empty else None,
                "max": control_df[var].max() if not control_df.empty else None,
                "n": len(control_df)
            }
        ]
        
        for i, group_df in enumerate(exp_dfs):
            group_stats.append({
                "group": f"Experimental Group {i+1}",
                "mean": group_df[var].mean() if not group_df.empty else None,
                "std": group_df[var].std() if not group_df.empty else None,
                "min": group_df[var].min() if not group_df.empty else None,
                "max": group_df[var].max() if not group_df.empty else None,
                "n": len(group_df)
            })
        
        summary[var] = group_stats
    
    return {
        "t_tests": test_results,
        "anova": anova_results,
        "summary": summary,
        "combined_data": combined_df
    }

# Sidebar for navigation
st.sidebar.header("Experiment Navigation")

# Display different experiment stages
if st.session_state.experiment_stage == "design":
    experiment_tab = st.sidebar.radio(
        "Design Process",
        ["Experiment Setup", "Variable Selection", "Group Configuration", "Confirmation"]
    )
    
    if experiment_tab == "Experiment Setup":
        st.header("1. Experiment Setup")
        st.subheader("Basic Information")
        
        experiment_name = st.text_input(
            "Experiment Name", 
            value=st.session_state.experiment_design["name"]
        )
        
        experiment_description = st.text_area(
            "Experiment Description",
            value=st.session_state.experiment_design["description"],
            placeholder="Describe the purpose and context of this experiment..."
        )
        
        experiment_hypothesis = st.text_area(
            "Research Hypothesis",
            value=st.session_state.experiment_design["hypothesis"],
            placeholder="Formulate your research hypothesis in a clear, testable statement..."
        )
        
        # Update session state
        st.session_state.experiment_design["name"] = experiment_name
        st.session_state.experiment_design["description"] = experiment_description
        st.session_state.experiment_design["hypothesis"] = experiment_hypothesis
        
        st.subheader("Experiment Settings")
        
        replications = st.number_input(
            "Number of Replications per Group",
            min_value=1,
            max_value=20,
            value=st.session_state.experiment_design["replications"],
            help="How many times each condition will be repeated"
        )
        
        randomization = st.checkbox(
            "Use Randomization",
            value=st.session_state.experiment_design["randomization"],
            help="Randomize simulation parameters within defined ranges"
        )
        
        # Update session state
        st.session_state.experiment_design["replications"] = replications
        st.session_state.experiment_design["randomization"] = randomization
        
        st.markdown("---")
        st.write("Click 'Next' to continue to variable selection.")
        if st.button("Next", key="next_to_variables"):
            st.sidebar.radio("Design Process", ["Experiment Setup", "Variable Selection", "Group Configuration", "Confirmation"], index=1)
            st.rerun()
    
    elif experiment_tab == "Variable Selection":
        st.header("2. Variable Selection")
        
        st.subheader("Independent Variable")
        st.write("Select one parameter to vary across experimental groups:")
        
        independent_var = st.selectbox(
            "Independent Variable",
            options=["grid_size", "resources", "dangers", "goals", "steps"],
            index=0 if not st.session_state.experiment_design["independent_variable"] else 
                  ["grid_size", "resources", "dangers", "goals", "steps"].index(
                      st.session_state.experiment_design["independent_variable"].get("name", "grid_size")
                  )
        )
        
        # Define variable levels based on selected parameter
        st.write("Define levels for your independent variable:")
        
        if independent_var == "grid_size":
            var_control = st.number_input("Control Group Value", min_value=5, max_value=20, value=10)
            var_levels = st.multiselect(
                "Experimental Group Values",
                options=[5, 8, 10, 12, 15, 20],
                default=[5, 15, 20] if var_control not in [5, 15, 20] else [8, 12, var_control]
            )
            var_name = "Grid Size"
            var_unit = "cells"
        
        elif independent_var == "resources":
            var_control = st.number_input("Control Group Value", min_value=0, max_value=30, value=10)
            var_levels = st.slider("Experimental Group Values", min_value=0, max_value=30, value=(5, 15, 25))
            var_name = "Resources"
            var_unit = "count"
        
        elif independent_var == "dangers":
            var_control = st.number_input("Control Group Value", min_value=0, max_value=20, value=5)
            var_levels = st.slider("Experimental Group Values", min_value=0, max_value=20, value=(0, 10, 20))
            var_name = "Dangers"
            var_unit = "count"
        
        elif independent_var == "goals":
            var_control = st.number_input("Control Group Value", min_value=0, max_value=5, value=1)
            var_levels = st.multiselect(
                "Experimental Group Values",
                options=[0, 1, 2, 3, 4, 5],
                default=[0, 2, 3] if var_control not in [0, 2, 3] else [1, 4, 5]
            )
            var_name = "Goals"
            var_unit = "count"
        
        elif independent_var == "steps":
            var_control = st.number_input("Control Group Value", min_value=50, max_value=500, value=100)
            var_levels = st.multiselect(
                "Experimental Group Values",
                options=[50, 100, 200, 300, 400, 500],
                default=[50, 200, 500] if var_control not in [50, 200, 500] else [100, 300, 400]
            )
            var_name = "Simulation Steps"
            var_unit = "steps"
        
        # Update session state for independent variable
        st.session_state.experiment_design["independent_variable"] = {
            "name": independent_var,
            "display_name": var_name,
            "unit": var_unit,
            "control_value": var_control,
            "experimental_values": var_levels
        }
        
        st.subheader("Dependent Variables")
        st.write("Select metrics to measure as dependent variables:")
        
        dependent_vars = st.multiselect(
            "Dependent Variables",
            options=[
                "relationship_score", "avg_resonance", "max_resonance", 
                "avg_bond_strength_a", "avg_bond_strength_b",
                "A_phi_mean", "A_sigma_mean", "A_delta_mean",
                "B_phi_mean", "B_sigma_mean", "B_delta_mean",
                "time_together"
            ],
            default=st.session_state.experiment_design["dependent_variables"] or ["relationship_score", "avg_resonance", "time_together"],
            format_func=lambda x: {
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
            }.get(x, x)
        )
        
        # Update session state
        st.session_state.experiment_design["dependent_variables"] = dependent_vars
        
        st.markdown("---")
        st.write("Click 'Next' to continue to group configuration.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Back", key="back_to_setup"):
                st.sidebar.radio("Design Process", ["Experiment Setup", "Variable Selection", "Group Configuration", "Confirmation"], index=0)
                st.rerun()
        with col2:
            if st.button("Next", key="next_to_groups"):
                st.sidebar.radio("Design Process", ["Experiment Setup", "Variable Selection", "Group Configuration", "Confirmation"], index=2)
                st.rerun()
    
    elif experiment_tab == "Group Configuration":
        st.header("3. Group Configuration")
        
        # Check if we have independent variable defined
        if not st.session_state.experiment_design["independent_variable"]:
            st.error("Please define your independent variable first.")
            if st.button("Go to Variable Selection"):
                st.sidebar.radio("Design Process", ["Experiment Setup", "Variable Selection", "Group Configuration", "Confirmation"], index=1)
                st.rerun()
        else:
            # Display independent variable information
            ind_var = st.session_state.experiment_design["independent_variable"]
            st.subheader(f"Independent Variable: {ind_var['display_name']}")
            st.write(f"Control value: {ind_var['control_value']} {ind_var['unit']}")
            st.write(f"Experimental values: {', '.join([str(v) for v in ind_var['experimental_values']])} {ind_var['unit']}")
            
            st.subheader("Control Group Configuration")
            
            # Default parameter values
            default_params = {
                "grid_size": 10,
                "resources": 10,
                "dangers": 5,
                "goals": 1,
                "steps": 100
            }
            
            # Set the independent variable value for control group
            default_params[ind_var["name"]] = ind_var["control_value"]
            
            # Control group configuration
            with st.expander("Control Group Parameters", expanded=True):
                # Create dictionary to store control group parameters
                control_params = {}
                
                for param, default in default_params.items():
                    # Skip the independent variable as it's already set
                    if param == ind_var["name"]:
                        st.write(f"**{param.capitalize()}:** {ind_var['control_value']} (independent variable)")
                        control_params[param] = ind_var["control_value"]
                        continue
                    
                    # For other parameters, allow configuration
                    if param in st.session_state.experiment_design["control_group"]:
                        default = st.session_state.experiment_design["control_group"][param]
                    
                    if param == "grid_size":
                        control_params[param] = st.number_input(
                            "Grid Size",
                            min_value=5,
                            max_value=20,
                            value=default
                        )
                    elif param == "resources":
                        control_params[param] = st.number_input(
                            "Resources",
                            min_value=0,
                            max_value=30,
                            value=default
                        )
                    elif param == "dangers":
                        control_params[param] = st.number_input(
                            "Dangers",
                            min_value=0,
                            max_value=20,
                            value=default
                        )
                    elif param == "goals":
                        control_params[param] = st.number_input(
                            "Goals",
                            min_value=0,
                            max_value=5,
                            value=default
                        )
                    elif param == "steps":
                        control_params[param] = st.number_input(
                            "Steps",
                            min_value=50,
                            max_value=500,
                            value=default
                        )
                
                # Agent positions
                agent_positions = st.checkbox(
                    "Set fixed agent positions?",
                    value=st.session_state.experiment_design.get("control_group", {}).get("use_fixed_positions", False)
                )
                
                if agent_positions:
                    col1, col2 = st.columns(2)
                    with col1:
                        a_x = st.number_input(
                            "Agent A X",
                            min_value=0,
                            max_value=control_params.get("grid_size", 10) - 1,
                            value=st.session_state.experiment_design.get("control_group", {}).get("agent_a_x", 0)
                        )
                        a_y = st.number_input(
                            "Agent A Y",
                            min_value=0,
                            max_value=control_params.get("grid_size", 10) - 1,
                            value=st.session_state.experiment_design.get("control_group", {}).get("agent_a_y", 0)
                        )
                    with col2:
                        b_x = st.number_input(
                            "Agent B X",
                            min_value=0,
                            max_value=control_params.get("grid_size", 10) - 1,
                            value=st.session_state.experiment_design.get("control_group", {}).get("agent_b_x", 5)
                        )
                        b_y = st.number_input(
                            "Agent B Y",
                            min_value=0,
                            max_value=control_params.get("grid_size", 10) - 1,
                            value=st.session_state.experiment_design.get("control_group", {}).get("agent_b_y", 5)
                        )
                    
                    control_params["use_fixed_positions"] = True
                    control_params["agent_a_x"] = a_x
                    control_params["agent_a_y"] = a_y
                    control_params["agent_b_x"] = b_x
                    control_params["agent_b_y"] = b_y
                    control_params["start_pos_a"] = (a_x, a_y)
                    control_params["start_pos_b"] = (b_x, b_y)
                else:
                    control_params["use_fixed_positions"] = False
                    control_params["start_pos_a"] = (None, None)
                    control_params["start_pos_b"] = (None, None)
            
            # Update session state for control group
            st.session_state.experiment_design["control_group"] = control_params
            
            st.subheader("Experimental Groups Configuration")
            
            # Create experimental groups based on independent variable values
            experimental_values = ind_var["experimental_values"]
            experimental_groups = []
            
            for i, value in enumerate(experimental_values):
                with st.expander(f"Experimental Group {i+1}: {ind_var['display_name']} = {value} {ind_var['unit']}", expanded=True):
                    # Start with control group parameters
                    group_params = control_params.copy()
                    
                    # Override independent variable
                    group_params[ind_var["name"]] = value
                    
                    # Display parameters
                    st.write("Parameters (other than independent variable are inherited from control group):")
                    for param, val in group_params.items():
                        if param not in ["use_fixed_positions", "agent_a_x", "agent_a_y", "agent_b_x", "agent_b_y", "start_pos_a", "start_pos_b"]:
                            if param == ind_var["name"]:
                                st.write(f"**{param.capitalize()}:** {val} (manipulated)")
                            else:
                                st.write(f"{param.capitalize()}: {val}")
                    
                    # Agent positions
                    if group_params.get("use_fixed_positions", False):
                        st.write(f"Agent A Position: ({group_params['agent_a_x']}, {group_params['agent_a_y']})")
                        st.write(f"Agent B Position: ({group_params['agent_b_x']}, {group_params['agent_b_y']})")
                    else:
                        st.write("Agent Positions: Random")
                    
                    # Add to experimental groups
                    experimental_groups.append(group_params)
            
            # Update session state for experimental groups
            st.session_state.experiment_design["experimental_groups"] = experimental_groups
            
            st.markdown("---")
            st.write("Click 'Next' to review and confirm your experiment design.")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Back", key="back_to_variables"):
                    st.sidebar.radio("Design Process", ["Experiment Setup", "Variable Selection", "Group Configuration", "Confirmation"], index=1)
                    st.rerun()
            with col2:
                if st.button("Next", key="next_to_confirmation"):
                    st.sidebar.radio("Design Process", ["Experiment Setup", "Variable Selection", "Group Configuration", "Confirmation"], index=3)
                    st.rerun()
    
    elif experiment_tab == "Confirmation":
        st.header("4. Experiment Confirmation")
        
        # Check if all necessary components are defined
        if (not st.session_state.experiment_design["name"] or
            not st.session_state.experiment_design["hypothesis"] or
            not st.session_state.experiment_design["independent_variable"] or
            not st.session_state.experiment_design["control_group"] or
            not st.session_state.experiment_design["experimental_groups"] or
            not st.session_state.experiment_design["dependent_variables"]):
            
            st.error("Please complete all sections of the experiment design before confirming.")
            missing_sections = []
            
            if not st.session_state.experiment_design["name"]:
                missing_sections.append("Experiment name")
            if not st.session_state.experiment_design["hypothesis"]:
                missing_sections.append("Research hypothesis")
            if not st.session_state.experiment_design["independent_variable"]:
                missing_sections.append("Independent variable")
            if not st.session_state.experiment_design["control_group"]:
                missing_sections.append("Control group configuration")
            if not st.session_state.experiment_design["experimental_groups"]:
                missing_sections.append("Experimental groups configuration")
            if not st.session_state.experiment_design["dependent_variables"]:
                missing_sections.append("Dependent variables")
            
            st.write(f"Missing sections: {', '.join(missing_sections)}")
            
            if st.button("Go to Experiment Setup"):
                st.sidebar.radio("Design Process", ["Experiment Setup", "Variable Selection", "Group Configuration", "Confirmation"], index=0)
                st.rerun()
        else:
            # Display experiment summary
            st.subheader("Experiment Summary")
            
            design = st.session_state.experiment_design
            ind_var = design["independent_variable"]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Name:** {design['name']}")
                st.write(f"**Hypothesis:** {design['hypothesis']}")
                st.write(f"**Replications:** {design['replications']} per group")
                st.write(f"**Randomization:** {'Enabled' if design['randomization'] else 'Disabled'}")
            
            with col2:
                st.write(f"**Independent Variable:** {ind_var['display_name']}")
                st.write(f"**Control Value:** {ind_var['control_value']} {ind_var['unit']}")
                st.write(f"**Experimental Values:** {', '.join([str(v) for v in ind_var['experimental_values']])} {ind_var['unit']}")
                
                dependent_var_names = {
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
                
                st.write(f"**Dependent Variables:** {', '.join([dependent_var_names.get(v, v) for v in design['dependent_variables']])}")
            
            # Group summary
            st.subheader("Group Configuration")
            
            # Control group
            with st.expander("Control Group", expanded=True):
                st.write(f"**{ind_var['name'].capitalize()}:** {ind_var['control_value']} {ind_var['unit']} (control value)")
                
                for param, value in design["control_group"].items():
                    if param not in ["use_fixed_positions", "agent_a_x", "agent_a_y", "agent_b_x", "agent_b_y", "start_pos_a", "start_pos_b"] and param != ind_var["name"]:
                        st.write(f"**{param.capitalize()}:** {value}")
                
                if design["control_group"].get("use_fixed_positions", False):
                    st.write(f"**Agent A Position:** ({design['control_group']['agent_a_x']}, {design['control_group']['agent_a_y']})")
                    st.write(f"**Agent B Position:** ({design['control_group']['agent_b_x']}, {design['control_group']['agent_b_y']})")
                else:
                    st.write("**Agent Positions:** Random")
            
            # Experimental groups
            for i, group in enumerate(design["experimental_groups"]):
                with st.expander(f"Experimental Group {i+1}", expanded=True):
                    st.write(f"**{ind_var['name'].capitalize()}:** {group[ind_var['name']]} {ind_var['unit']} (experimental value)")
                    
                    for param, value in group.items():
                        if param not in ["use_fixed_positions", "agent_a_x", "agent_a_y", "agent_b_x", "agent_b_y", "start_pos_a", "start_pos_b"] and param != ind_var["name"]:
                            st.write(f"**{param.capitalize()}:** {value}")
                    
                    if group.get("use_fixed_positions", False):
                        st.write(f"**Agent A Position:** ({group['agent_a_x']}, {group['agent_a_y']})")
                        st.write(f"**Agent B Position:** ({group['agent_b_x']}, {group['agent_b_y']})")
                    else:
                        st.write("**Agent Positions:** Random")
            
            # Total number of simulations
            total_sims = design["replications"] * (1 + len(design["experimental_groups"]))
            st.info(f"This experiment will run a total of {total_sims} simulations ({design['replications']} replications × {1 + len(design['experimental_groups'])} groups).")
            
            # Estimated runtime
            avg_sim_time = 0.5  # Estimated seconds per simulation
            est_runtime = total_sims * avg_sim_time
            st.write(f"Estimated runtime: {est_runtime:.1f} seconds")
            
            # Execute experiment button
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Edit Experiment", key="back_to_groups"):
                    st.sidebar.radio("Design Process", ["Experiment Setup", "Variable Selection", "Group Configuration", "Confirmation"], index=2)
                    st.rerun()
            with col2:
                if st.button("Run Experiment", type="primary", key="run_experiment"):
                    # Transition to running stage
                    st.session_state.experiment_stage = "running"
                    st.rerun()

elif st.session_state.experiment_stage == "running":
    st.header("Experiment Execution")
    
    # Display experiment info
    design = st.session_state.experiment_design
    
    st.subheader(design["name"])
    st.write(f"**Hypothesis:** {design['hypothesis']}")
    
    # Setup progress tracking
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    # Execute the experiment
    control_group = design["control_group"]
    experimental_groups = design["experimental_groups"]
    replications = design["replications"]
    
    # Prepare control group parameters for simulation
    control_params = {k: v for k, v in control_group.items() 
                     if k in ["grid_size", "resources", "dangers", "goals", "steps", "start_pos_a", "start_pos_b"]}
    
    # Prepare experimental group parameters
    experimental_params = []
    for group in experimental_groups:
        params = {k: v for k, v in group.items() 
                 if k in ["grid_size", "resources", "dangers", "goals", "steps", "start_pos_a", "start_pos_b"]}
        experimental_params.append(params)
    
    # Total number of simulations
    total_sims = replications * (1 + len(experimental_groups))
    sims_completed = 0
    
    # Run control group simulations
    progress_text.text(f"Running control group simulations...")
    
    # Umschließen Sie Fortschrittsfunktionen mit einer Klasse, um Nonlocal-Probleme zu vermeiden
    class ProgressTracker:
        def __init__(self, total):
            self.completed = 0
            self.total = total
            
        def update(self, message):
            self.completed += 1
            progress_text.text(message)
            progress_bar.progress(self.completed / self.total)
    
    # Instanz des Trackers erstellen
    progress_tracker = ProgressTracker(total_sims)
    
    # Run the control group
    control_results = run_experiment_group(
        control_params, 
        replications, 
        "Control Group",
        progress_callback=progress_tracker.update
    )
    
    # Run experimental groups
    experimental_results = []
    for i, params in enumerate(experimental_params):
        group_results = run_experiment_group(
            params, 
            replications, 
            f"Experimental Group {i+1}",
            progress_callback=progress_tracker.update
        )
        experimental_results.append(group_results)
    
    # Store results in session state
    st.session_state.experiment_results = {
        "control_data": control_results,
        "experimental_data": experimental_results
    }
    
    # Analyze results
    progress_text.text("Analyzing experiment results...")
    analysis_results = analyze_experiment_results(
        control_results,
        experimental_results,
        design["dependent_variables"],
        design["independent_variable"]
    )
    
    # Store analysis in session state
    st.session_state.experiment_results["analysis"] = analysis_results
    
    # Clear progress indicators and transition to analysis stage
    progress_text.empty()
    progress_bar.empty()
    
    st.success("Experiment completed successfully!")
    st.session_state.experiment_stage = "analysis"
    st.rerun()

elif st.session_state.experiment_stage == "analysis":
    st.header("Experiment Results Analysis")
    
    # Get experiment design and results
    design = st.session_state.experiment_design
    results = st.session_state.experiment_results
    analysis = results.get("analysis", {})
    
    # Display experiment information
    st.subheader(design["name"])
    st.write(f"**Hypothesis:** {design['hypothesis']}")
    
    # Analysis tabs
    analysis_tab = st.radio(
        "Analysis Sections",
        ["Hypothesis Testing", "Detailed Analysis", "Visual Comparison", "Export Results"],
        horizontal=True
    )
    
    if analysis_tab == "Hypothesis Testing":
        st.subheader("Hypothesis Test Results")
        
        # Get t-test results for the dependent variables
        t_tests = analysis.get("t_tests", {})
        
        if t_tests:
            # Get independent variable info
            ind_var = design["independent_variable"]
            
            for i, dv in enumerate(design["dependent_variables"]):
                # Get friendly name for dependent variable
                dv_name = {
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
                }.get(dv, dv)
                
                st.write(f"### Effect on {dv_name}")
                
                for j, test in enumerate(t_tests.get(dv, [])):
                    exp_value = design["experimental_groups"][j][ind_var["name"]]
                    
                    if "error" in test:
                        st.error(f"Error in test for Experimental Group {j+1} ({ind_var['display_name']} = {exp_value}): {test['error']}")
                        continue
                    
                    # Create a visual indicator of significance
                    if test["significant"]:
                        st.success(f"✅ **Significant effect found** when {ind_var['display_name']} = {exp_value} {ind_var['unit']}")
                    else:
                        st.info(f"❌ No significant effect when {ind_var['display_name']} = {exp_value} {ind_var['unit']}")
                    
                    # Display detailed statistics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Control Group Mean", 
                            f"{test['control_mean']:.3f}"
                        )
                    
                    with col2:
                        st.metric(
                            f"Experimental Group Mean",
                            f"{test['experimental_mean']:.3f}",
                            delta=f"{test['mean_difference']:.3f}" if test['mean_difference'] != 0 else "0"
                        )
                    
                    with col3:
                        st.metric(
                            "Effect Size (Cohen's d)",
                            f"{test['effect_size']:.3f}",
                            delta=test['effect_size_interpretation']
                        )
                    
                    # T-test details
                    st.write(f"t({test.get('df', 'N/A')}) = {test['t_statistic']:.3f}, p = {test['p_value']:.4f}")
                    
                    # Interpretation
                    if test["significant"]:
                        direction = "increased" if test["mean_difference"] > 0 else "decreased"
                        st.write(f"**Interpretation:** When {ind_var['display_name']} was changed from {ind_var['control_value']} to {exp_value} {ind_var['unit']}, {dv_name} {direction} by {abs(test['mean_difference_percent']):.1f}%. This effect is statistically significant with {test['effect_size_interpretation']} practical significance.")
                    else:
                        st.write(f"**Interpretation:** Changing {ind_var['display_name']} from {ind_var['control_value']} to {exp_value} {ind_var['unit']} did not have a statistically significant effect on {dv_name}.")
                    
                    st.markdown("---")
                
                # ANOVA results if available
                anova = analysis.get("anova", {}).get(dv, {})
                if anova and "error" not in anova:
                    st.write(f"**ANOVA for all groups:** F = {anova['f_statistic']:.3f}, p = {anova['p_value']:.4f}")
                    
                    if anova["significant"]:
                        st.success(f"Overall, {ind_var['display_name']} has a significant effect on {dv_name}.")
                    else:
                        st.info(f"Overall, {ind_var['display_name']} does not have a significant effect on {dv_name}.")
                
                st.markdown("---")
            
            # Overall conclusion
            st.subheader("Conclusion")
            
            # Count significant results
            sig_count = 0
            total_tests = 0
            
            for dv in design["dependent_variables"]:
                for test in t_tests.get(dv, []):
                    if "significant" in test:
                        total_tests += 1
                        if test["significant"]:
                            sig_count += 1
            
            if sig_count > 0:
                if sig_count == total_tests:
                    st.success(f"**Strong support for hypothesis.** All {sig_count} tests showed significant effects.")
                elif sig_count >= total_tests // 2:
                    st.success(f"**Partial support for hypothesis.** {sig_count} out of {total_tests} tests showed significant effects.")
                else:
                    st.info(f"**Limited support for hypothesis.** Only {sig_count} out of {total_tests} tests showed significant effects.")
            else:
                st.warning("**No support for hypothesis.** None of the tests showed significant effects.")
            
            # Relate back to original hypothesis
            st.write(f"**Original Hypothesis:** {design['hypothesis']}")
            
            # Suggestion for further research
            st.write("**Suggestions for further research:**")
            if sig_count == 0:
                st.write("- Consider testing different levels of the independent variable")
                st.write("- Explore other potential factors that might influence agent behavior")
                st.write("- Increase the number of replications to improve statistical power")
            elif sig_count < total_tests:
                st.write("- Focus future studies on the specific dependent variables that showed significant effects")
                st.write("- Explore non-linear relationships or threshold effects")
                st.write("- Consider interaction effects with other variables")
            else:
                st.write("- Investigate the mechanism behind the observed effects")
                st.write("- Test if the relationship is linear or has optimal values")
                st.write("- Explore interactions with other variables")
        else:
            st.error("No analysis results available. Please run the experiment again.")
    
    elif analysis_tab == "Detailed Analysis":
        st.subheader("Detailed Statistical Analysis")
        
        # Get summary statistics
        summary = analysis.get("summary", {})
        
        if summary:
            # Select dependent variable to analyze
            dv = st.selectbox(
                "Select Dependent Variable",
                options=design["dependent_variables"],
                format_func=lambda x: {
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
                }.get(x, x)
            )
            
            # Display summary statistics for this dependent variable
            if dv in summary:
                stats = summary[dv]
                
                # Create a DataFrame for display
                stats_df = pd.DataFrame(stats)
                
                # Round numeric columns
                for col in stats_df.columns:
                    if col not in ["group", "n"] and stats_df[col].dtype in ["float64", "float32"]:
                        stats_df[col] = stats_df[col].round(3)
                
                st.dataframe(stats_df, use_container_width=True)
                
                # Detailed statistical tests
                st.subheader("Statistical Tests")
                
                # T-tests
                t_tests = analysis.get("t_tests", {}).get(dv, [])
                if t_tests:
                    st.write("**T-tests (Control vs. Experimental Groups)**")
                    
                    t_test_df = pd.DataFrame([
                        {
                            "Group": test["group_name"],
                            "t-statistic": round(test["t_statistic"], 3),
                            "p-value": round(test["p_value"], 4),
                            "Mean Difference": round(test["mean_difference"], 3),
                            "% Change": f"{round(test['mean_difference_percent'], 1)}%",
                            "Effect Size": round(test["effect_size"], 3),
                            "Significance": "Yes" if test["significant"] else "No"
                        }
                        for test in t_tests if "error" not in test
                    ])
                    
                    st.dataframe(t_test_df, use_container_width=True)
                
                # ANOVA
                anova = analysis.get("anova", {}).get(dv, {})
                if anova and "error" not in anova:
                    st.write("**ANOVA (All Groups)**")
                    
                    anova_df = pd.DataFrame([{
                        "F-statistic": round(anova["f_statistic"], 3),
                        "p-value": round(anova["p_value"], 4),
                        "Significance": "Yes" if anova["significant"] else "No"
                    }])
                    
                    st.dataframe(anova_df, use_container_width=True)
                
                # Display raw data points for this variable
                st.subheader("Individual Data Points")
                
                # Combine data from all groups
                all_data = []
                
                # Control group
                for d in results["control_data"]:
                    all_data.append({
                        "Group": "Control",
                        "Value": d[dv],
                        f"{design['independent_variable']['name']}": design["independent_variable"]["control_value"]
                    })
                
                # Experimental groups
                for i, group_data in enumerate(results["experimental_data"]):
                    exp_value = design["experimental_groups"][i][design["independent_variable"]["name"]]
                    for d in group_data:
                        all_data.append({
                            "Group": f"Experimental {i+1}",
                            "Value": d[dv],
                            f"{design['independent_variable']['name']}": exp_value
                        })
                
                # Convert to DataFrame
                df = pd.DataFrame(all_data)
                
                # Plot individual data points
                fig = px.strip(
                    df, 
                    x=f"{design['independent_variable']['name']}", 
                    y="Value",
                    color="Group",
                    title=f"Individual Data Points for {dv}",
                    labels={"Value": dv}
                )
                
                # Add mean points
                group_means = df.groupby([f"{design['independent_variable']['name']}", "Group"])["Value"].mean().reset_index()
                fig.add_trace(
                    go.Scatter(
                        x=group_means[f"{design['independent_variable']['name']}"],
                        y=group_means["Value"],
                        mode="markers",
                        marker=dict(size=12, symbol="x", color="black"),
                        name="Group Mean"
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No summary statistics available. Please run the experiment again.")
    
    elif analysis_tab == "Visual Comparison":
        st.subheader("Visual Comparison of Results")
        
        # Get data for visualization
        if results["control_data"] and results["experimental_data"]:
            # Combine data from all groups
            all_data = []
            
            # Control group
            ind_var_name = design["independent_variable"]["name"]
            ind_var_value = design["independent_variable"]["control_value"]
            
            for d in results["control_data"]:
                d_copy = {k: v for k, v in d.items() if k != "simulation_data" and k != "parameters"}
                d_copy["Group"] = "Control"
                d_copy[ind_var_name] = ind_var_value
                all_data.append(d_copy)
            
            # Experimental groups
            for i, group_data in enumerate(results["experimental_data"]):
                exp_value = design["experimental_groups"][i][ind_var_name]
                for d in group_data:
                    d_copy = {k: v for k, v in d.items() if k != "simulation_data" and k != "parameters"}
                    d_copy["Group"] = f"Experimental {i+1}"
                    d_copy[ind_var_name] = exp_value
                    all_data.append(d_copy)
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            
            # Visualization types
            viz_type = st.radio(
                "Visualization Type",
                ["Bar Chart", "Box Plot", "Violin Plot", "Scatter Plot"],
                horizontal=True
            )
            
            # Select dependent variables to visualize
            dvs = st.multiselect(
                "Select Dependent Variables",
                options=design["dependent_variables"],
                default=design["dependent_variables"][:min(3, len(design["dependent_variables"]))],
                format_func=lambda x: {
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
                }.get(x, x)
            )
            
            if dvs:
                for dv in dvs:
                    dv_name = {
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
                    }.get(dv, dv)
                    
                    if viz_type == "Bar Chart":
                        # Group by independent variable and calculate mean
                        group_means = df.groupby(ind_var_name)[dv].mean().reset_index()
                        group_errors = df.groupby(ind_var_name)[dv].sem().reset_index()
                        
                        # Merge means and errors
                        plot_data = pd.merge(group_means, group_errors, on=ind_var_name, suffixes=('_mean', '_sem'))
                        
                        fig = px.bar(
                            plot_data,
                            x=ind_var_name,
                            y=f"{dv}_mean",
                            error_y=f"{dv}_sem",
                            title=f"Effect of {design['independent_variable']['display_name']} on {dv_name}",
                            labels={
                                ind_var_name: design['independent_variable']['display_name'],
                                f"{dv}_mean": dv_name
                            }
                        )
                        
                        # Add individual data points
                        fig.add_trace(
                            go.Scatter(
                                x=df[ind_var_name],
                                y=df[dv],
                                mode="markers",
                                marker=dict(size=8, opacity=0.6),
                                name="Individual Runs"
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif viz_type == "Box Plot":
                        fig = px.box(
                            df,
                            x=ind_var_name,
                            y=dv,
                            title=f"Distribution of {dv_name} by {design['independent_variable']['display_name']}",
                            labels={
                                ind_var_name: design['independent_variable']['display_name'],
                                dv: dv_name
                            }
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif viz_type == "Violin Plot":
                        fig = px.violin(
                            df,
                            x=ind_var_name,
                            y=dv,
                            box=True,
                            points="all",
                            title=f"Distribution of {dv_name} by {design['independent_variable']['display_name']}",
                            labels={
                                ind_var_name: design['independent_variable']['display_name'],
                                dv: dv_name
                            }
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif viz_type == "Scatter Plot":
                        fig = px.scatter(
                            df,
                            x=ind_var_name,
                            y=dv,
                            color="Group",
                            trendline="ols",
                            title=f"Relationship between {design['independent_variable']['display_name']} and {dv_name}",
                            labels={
                                ind_var_name: design['independent_variable']['display_name'],
                                dv: dv_name
                            }
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                # Relationship between dependent variables
                if len(dvs) >= 2:
                    st.subheader("Relationships Between Dependent Variables")
                    
                    dv1 = st.selectbox("First Variable", options=dvs, index=0)
                    dv2 = st.selectbox("Second Variable", options=[dv for dv in dvs if dv != dv1], index=0)
                    
                    dv1_name = {
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
                    }.get(dv1, dv1)
                    
                    dv2_name = {
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
                    }.get(dv2, dv2)
                    
                    if dv1 != dv2:
                        fig = px.scatter(
                            df,
                            x=dv1,
                            y=dv2,
                            color=ind_var_name,
                            trendline="ols",
                            title=f"Relationship between {dv1_name} and {dv2_name}",
                            labels={
                                dv1: dv1_name,
                                dv2: dv2_name,
                                ind_var_name: design['independent_variable']['display_name']
                            }
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Calculate correlation
                        corr = df[dv1].corr(df[dv2])
                        st.metric("Correlation Coefficient", f"{corr:.3f}")
                        
                        # Correlation by group
                        st.write("Correlation by Independent Variable Value:")
                        
                        for val in df[ind_var_name].unique():
                            group_data = df[df[ind_var_name] == val]
                            group_corr = group_data[dv1].corr(group_data[dv2])
                            st.write(f"- {design['independent_variable']['display_name']} = {val}: r = {group_corr:.3f}")
            else:
                st.info("Please select at least one dependent variable to visualize.")
        else:
            st.error("No experiment data available. Please run the experiment again.")
    
    elif analysis_tab == "Export Results":
        st.subheader("Export Experiment Results")
        
        # Experiment summary
        st.write("### Experiment Summary")
        
        summary_text = f"""
        # Experiment Report: {design['name']}
        
        ## Hypothesis
        {design['hypothesis']}
        
        ## Experiment Design
        - **Independent Variable:** {design['independent_variable']['display_name']}
        - **Control Value:** {design['independent_variable']['control_value']} {design['independent_variable']['unit']}
        - **Experimental Values:** {', '.join([str(v) for v in design['independent_variable']['experimental_values']])} {design['independent_variable']['unit']}
        - **Dependent Variables:** {', '.join(design['dependent_variables'])}
        - **Replications:** {design['replications']} per group
        """
        
        # Add statistical results if available
        if "analysis" in results and "t_tests" in results["analysis"]:
            summary_text += "\n\n## Statistical Results\n"
            
            for dv in design["dependent_variables"]:
                dv_name = {
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
                }.get(dv, dv)
                
                summary_text += f"\n### Effect on {dv_name}\n"
                
                t_tests = results["analysis"]["t_tests"].get(dv, [])
                for i, test in enumerate(t_tests):
                    if "error" in test:
                        summary_text += f"- Error in test for Experimental Group {i+1}: {test['error']}\n"
                        continue
                    
                    exp_value = design["experimental_groups"][i][design["independent_variable"]["name"]]
                    sig_text = "Significant effect" if test["significant"] else "No significant effect"
                    
                    summary_text += f"- **{sig_text}** when {design['independent_variable']['display_name']} = {exp_value}\n"
                    summary_text += f"  - Control mean: {test['control_mean']:.3f}\n"
                    summary_text += f"  - Experimental mean: {test['experimental_mean']:.3f}\n"
                    summary_text += f"  - Mean difference: {test['mean_difference']:.3f} ({test['mean_difference_percent']:.1f}%)\n"
                    summary_text += f"  - t-statistic: {test['t_statistic']:.3f}, p-value: {test['p_value']:.4f}\n"
                    summary_text += f"  - Effect size: {test['effect_size']:.3f} ({test['effect_size_interpretation']})\n"
        
        # Display summary and provide download button
        st.text_area("Experiment Report", summary_text, height=300)
        
        st.download_button(
            label="Download Report (Markdown)",
            data=summary_text,
            file_name=f"{design['name'].replace(' ', '_')}_report.md",
            mime="text/markdown"
        )
        
        # Export raw data
        st.write("### Export Raw Data")
        
        if "analysis" in results and "combined_data" in results["analysis"]:
            combined_df = results["analysis"]["combined_data"]
            
            csv = combined_df.to_csv(index=False)
            st.download_button(
                label="Download Raw Data (CSV)",
                data=csv,
                file_name=f"{design['name'].replace(' ', '_')}_data.csv",
                mime="text/csv"
            )
        else:
            st.info("No combined data available for export.")
        
        # Option to start a new experiment
        st.markdown("---")
        if st.button("Start New Experiment", type="primary"):
            # Reset experiment state
            st.session_state.experiment_stage = "design"
            st.session_state.experiment_design = {
                "experiment_id": str(uuid.uuid4()),
                "name": f"Experiment {datetime.now().strftime('%Y%m%d-%H%M%S')}",
                "description": "",
                "hypothesis": "",
                "independent_variable": {},
                "control_group": {},
                "experimental_groups": [],
                "dependent_variables": [],
                "replications": 5,
                "randomization": True
            }
            st.session_state.experiment_results = {
                "control_data": None,
                "experimental_data": [],
                "summary": None,
                "statistical_tests": None
            }
            st.rerun()