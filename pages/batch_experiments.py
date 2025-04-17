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
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
import visualization
import analysis
import database
import batch_simulation

# Page configuration
st.set_page_config(
    page_title="Batch Experiments",
    page_icon="🧪",
    layout="wide"
)

# Title and description
st.title("Batch Experiments")
st.markdown("""
Run multiple simulations with varying parameters to systematically explore the parameter space
and test hypotheses about agent behavior and relationships.
""")

# Sidebar for configuration
st.sidebar.header("Experiment Configuration")

# Experiment name
experiment_name = st.sidebar.text_input(
    "Experiment Name",
    value=f"Experiment {datetime.now().strftime('%Y%m%d-%H%M%S')}",
    help="A name to identify this batch of experiments"
)

# Parameter space configuration
st.sidebar.subheader("Parameter Space")
st.sidebar.markdown("Define the range of parameters to explore:")

# Define parameter ranges
with st.sidebar.expander("Grid Parameters", expanded=True):
    # Grid size with multiple options
    grid_size_options = st.multiselect(
        "Grid Sizes",
        options=[5, 8, 10, 12, 15, 20],
        default=[10],
        help="Multiple grid sizes to test"
    )
    
    # Resources as a range
    resources_min, resources_max = st.slider(
        "Resources Range", 
        min_value=0, 
        max_value=30, 
        value=(5, 15),
        help="Min and max number of resources on the grid"
    )
    resources_step = st.number_input("Resources Step", min_value=1, value=5)
    
    # Dangers as a range
    dangers_min, dangers_max = st.slider(
        "Dangers Range", 
        min_value=0, 
        max_value=20, 
        value=(0, 10),
        help="Min and max number of dangers on the grid"
    )
    dangers_step = st.number_input("Dangers Step", min_value=1, value=5)
    
    # Goals as a range
    goals_min, goals_max = st.slider(
        "Goals Range", 
        min_value=0, 
        max_value=5, 
        value=(1, 3),
        help="Min and max number of goals on the grid"
    )
    goals_step = st.number_input("Goals Step", min_value=1, value=1)

with st.sidebar.expander("Simulation Parameters", expanded=True):
    # Steps as a range
    steps_options = st.multiselect(
        "Simulation Steps",
        options=[50, 100, 200, 300, 500],
        default=[100],
        help="Number of steps to run each simulation"
    )
    
    # Agent starting positions
    use_fixed_positions = st.checkbox("Use fixed agent positions", value=False)
    
    if use_fixed_positions:
        col1, col2 = st.columns(2)
        with col1:
            agent_a_x = st.number_input("Agent A X", min_value=0, max_value=max(grid_size_options)-1 if grid_size_options else 9, value=0)
            agent_a_y = st.number_input("Agent A Y", min_value=0, max_value=max(grid_size_options)-1 if grid_size_options else 9, value=0)
        with col2:
            agent_b_x = st.number_input("Agent B X", min_value=0, max_value=max(grid_size_options)-1 if grid_size_options else 9, value=5)
            agent_b_y = st.number_input("Agent B Y", min_value=0, max_value=max(grid_size_options)-1 if grid_size_options else 5, value=5)
    
# Hypothesis formulation
st.sidebar.subheader("Hypothesis Testing")
with st.sidebar.expander("Define Hypothesis", expanded=False):
    hypothesis_enabled = st.checkbox("Enable hypothesis testing", value=False)
    
    if hypothesis_enabled:
        hypothesis_text = st.text_area(
            "Hypothesis Statement",
            value="There is a positive correlation between grid size and relationship score.",
            help="Formulate your hypothesis in plain text"
        )
        
        test_type = st.selectbox(
            "Test Type",
            options=["correlation", "t-test", "regression"],
            index=0,
            help="Type of statistical test to perform"
        )
        
        if test_type == "correlation":
            var1 = st.selectbox(
                "First Variable",
                options=["param_grid_size", "param_resources", "param_dangers", "param_goals", 
                        "param_steps", "avg_resonance", "final_relationship_score", 
                        "time_together", "distance_traveled_A", "distance_traveled_B"],
                index=0
            )
            
            var2 = st.selectbox(
                "Second Variable",
                options=["param_grid_size", "param_resources", "param_dangers", "param_goals", 
                        "param_steps", "avg_resonance", "final_relationship_score", 
                        "time_together", "distance_traveled_A", "distance_traveled_B"],
                index=6
            )
            
            expected_outcome = st.selectbox(
                "Expected Correlation",
                options=["positive", "negative", "any"],
                index=0
            )
            
            significance = st.slider(
                "Significance Level (α)",
                min_value=0.01,
                max_value=0.10,
                value=0.05,
                step=0.01
            )
            
            variables = [var1, var2]
        else:
            # Placeholder for other test types
            variables = []
            expected_outcome = "any"
            significance = 0.05
            st.info("Advanced test types will be implemented in a future update.")

# Execution settings
st.sidebar.subheader("Execution Settings")
parallelism = st.sidebar.slider(
    "Parallel Simulations",
    min_value=1,
    max_value=8,
    value=4,
    help="Number of simulations to run in parallel"
)

save_to_db = st.sidebar.checkbox(
    "Save individual simulations",
    value=False,
    help="Save each simulation to the database for later analysis"
)

# Main content area
tab1, tab2, tab3 = st.tabs(["Configuration", "Results", "Analysis"])

with tab1:
    st.header("Experiment Configuration Summary")
    
    # Calculate total number of parameter combinations
    resources_range = list(range(resources_min, resources_max + 1, resources_step))
    dangers_range = list(range(dangers_min, dangers_max + 1, dangers_step))
    goals_range = list(range(goals_min, goals_max + 1, goals_step))
    
    total_combinations = (len(grid_size_options) * 
                        len(resources_range) * 
                        len(dangers_range) * 
                        len(goals_range) * 
                        len(steps_options))
    
    # Configuration summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Experiment Setup")
        st.metric("Experiment Name", experiment_name)
        st.metric("Total Simulations", total_combinations)
        st.metric("Parallel Workers", parallelism)
        
    with col2:
        st.subheader("Grid Parameters")
        st.write(f"Grid Sizes: {grid_size_options}")
        st.write(f"Resources: {resources_range}")
        st.write(f"Dangers: {dangers_range}")
        st.write(f"Goals: {goals_range}")
        
    with col3:
        st.subheader("Simulation Parameters")
        st.write(f"Steps: {steps_options}")
        if use_fixed_positions:
            st.write(f"Agent A Position: ({agent_a_x}, {agent_a_y})")
            st.write(f"Agent B Position: ({agent_b_x}, {agent_b_y})")
        else:
            st.write("Agent Positions: Random")
    
    # Hypothesis summary
    if hypothesis_enabled:
        st.subheader("Hypothesis")
        st.info(hypothesis_text)
        st.write(f"Test type: {test_type}")
        if test_type == "correlation":
            st.write(f"Variables: {var1} and {var2}")
            st.write(f"Expected correlation: {expected_outcome}")
            st.write(f"Significance level (α): {significance}")
    
    # Warning for large experiments
    if total_combinations > 100:
        st.warning(f"You are about to run {total_combinations} simulations. This may take a long time.")
    
    # Run experiment button
    run_button = st.button("Run Experiment", type="primary", use_container_width=True)

# Store results in session state
if "batch_results" not in st.session_state:
    st.session_state.batch_results = None
if "batch_metadata" not in st.session_state:
    st.session_state.batch_metadata = None
if "hypothesis_result" not in st.session_state:
    st.session_state.hypothesis_result = None

# Execute the batch simulation
if run_button:
    # Prepare parameter space
    parameter_space = {
        "grid_size": grid_size_options,
        "resources": resources_range,
        "dangers": dangers_range,
        "goals": goals_range,
        "steps": steps_options
    }
    
    # Add fixed positions if enabled
    if use_fixed_positions:
        parameter_space["start_pos_a"] = [(agent_a_x, agent_a_y)]
        parameter_space["start_pos_b"] = [(agent_b_x, agent_b_y)]
    
    # Create batch simulator
    simulator = batch_simulation.BatchSimulator()
    
    # Set parameter space
    total_sims = simulator.set_parameter_space(parameter_space)
    
    # Set hypothesis if enabled
    if hypothesis_enabled:
        simulator.set_hypothesis(
            hypothesis_text=hypothesis_text,
            test_type=test_type,
            variables=variables,
            expected_outcome=expected_outcome,
            threshold=significance
        )
    
    # Show progress
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    # Execute batch simulations with progress tracking
    with st.spinner("Running batch simulations..."):
        # Start time
        start_time = time.time()
        
        # Run batch
        results_df = simulator.run_batch(
            max_workers=parallelism,
            save_to_db=save_to_db,
            experiment_name=experiment_name
        )
        
        # Update progress during execution
        for i in range(100):
            progress_text.text(f"Simulation progress: {i+1}%")
            progress_bar.progress(i + 1)
            
            # Check if simulations are done
            if simulator.metadata["completed_simulations"] >= total_sims:
                break
                
            # Sleep briefly to allow for updates
            time.sleep(0.01)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Save results to session state
        st.session_state.batch_results = results_df
        st.session_state.batch_metadata = simulator.metadata
        
        # Test hypothesis if enabled
        if hypothesis_enabled:
            st.session_state.hypothesis_result = simulator.test_hypothesis()
    
    # Clear progress indicators
    progress_text.empty()
    progress_bar.empty()
    
    # Show completion message
    st.success(f"Batch experiment completed! {simulator.metadata['completed_simulations']} simulations run in {execution_time:.2f} seconds.")
    
    # Switch to results tab
    st.rerun()  # Force rerun to update UI

# Display results
with tab2:
    st.header("Batch Simulation Results")
    
    if st.session_state.batch_results is not None and not st.session_state.batch_results.empty:
        results_df = st.session_state.batch_results
        metadata = st.session_state.batch_metadata
        
        # Results summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Execution Summary")
            st.metric("Total Simulations", metadata["total_simulations"])
            st.metric("Completed Simulations", metadata["completed_simulations"])
            st.metric("Success Rate", f"{metadata['success_rate']:.1f}%")
            
        with col2:
            st.subheader("Performance")
            st.metric("Total Runtime", f"{metadata.get('total_runtime', 0):.2f} seconds")
            st.metric("Average Runtime", f"{metadata['avg_runtime']:.2f} seconds per simulation")
            
        with col3:
            st.subheader("Hypothesis Test")
            if st.session_state.hypothesis_result:
                hypothesis = st.session_state.hypothesis_result
                confirmed = hypothesis["confirmed"]
                st.write(f"Hypothesis: {hypothesis['text']}")
                
                if confirmed:
                    st.success("✅ Hypothesis supported by data")
                else:
                    st.error("❌ Hypothesis not supported by data")
                
                if hypothesis["test_type"] == "correlation" and hypothesis["result"]:
                    corr = hypothesis["result"]["correlation"]
                    st.metric("Correlation", f"{corr:.3f}")
                    st.metric("p-value", f"{hypothesis['result']['p_value']:.3f}")
        
        # Data table with pagination
        st.subheader("Results Table")
        st.dataframe(results_df, use_container_width=True)
        
        # Download link for results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name=f"{experiment_name}_results.csv",
            mime="text/csv"
        )
    else:
        st.info("No batch simulation results available. Run an experiment to see results here.")

# Analysis tools
with tab3:
    st.header("Batch Simulation Analysis")
    
    if st.session_state.batch_results is not None and not st.session_state.batch_results.empty:
        results_df = st.session_state.batch_results
        
        # Parameter analysis
        st.subheader("Parameter Effects Analysis")
        
        # Select parameters and metrics to analyze
        col1, col2 = st.columns(2)
        
        with col1:
            param_to_analyze = st.selectbox(
                "Parameter to Analyze",
                options=[col for col in results_df.columns if col.startswith("param_")],
                format_func=lambda x: x.replace("param_", "")
            )
        
        with col2:
            metric_to_analyze = st.selectbox(
                "Metric to Analyze",
                options=[col for col in results_df.columns if not col.startswith("param_") and col != "database_id"],
                index=0
            )
        
        # Create visualization based on parameter type
        unique_values = results_df[param_to_analyze].nunique()
        
        if unique_values <= 10:  # Categorical/discrete parameter
            # Bar chart or box plot
            plot_type = st.radio(
                "Plot Type",
                options=["Bar Chart", "Box Plot"],
                index=0,
                horizontal=True
            )
            
            if plot_type == "Bar Chart":
                # Group by parameter and calculate mean of metric
                grouped_data = results_df.groupby(param_to_analyze)[metric_to_analyze].mean().reset_index()
                
                fig = px.bar(
                    grouped_data,
                    x=param_to_analyze,
                    y=metric_to_analyze,
                    title=f"Effect of {param_to_analyze.replace('param_', '')} on {metric_to_analyze}",
                    labels={
                        param_to_analyze: param_to_analyze.replace("param_", ""),
                        metric_to_analyze: metric_to_analyze
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show correlation coefficient
                corr = results_df[param_to_analyze].corr(results_df[metric_to_analyze])
                st.metric("Correlation Coefficient", f"{corr:.3f}")
                
                # Simple linear regression for trend analysis
                if len(results_df) > 1:
                    x = results_df[param_to_analyze].values
                    y = results_df[metric_to_analyze].values
                    
                    # Check if all x values are identical
                    if np.std(x) == 0 or len(np.unique(x)) <= 1:
                        st.warning(f"Cannot perform linear regression: all {param_to_analyze.replace('param_', '')} values are identical.")
                    else:
                        try:
                            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                            
                            st.write(f"Linear regression: {metric_to_analyze} = {slope:.3f} × {param_to_analyze.replace('param_', '')} + {intercept:.3f}")
                            st.write(f"R²: {r_value**2:.3f}, p-value: {p_value:.3f}")
                            
                            # Interpretation
                            if p_value < 0.05:
                                if slope > 0:
                                    st.success(f"There is a significant positive effect of {param_to_analyze.replace('param_', '')} on {metric_to_analyze}.")
                                else:
                                    st.success(f"There is a significant negative effect of {param_to_analyze.replace('param_', '')} on {metric_to_analyze}.")
                            else:
                                st.info(f"No significant effect of {param_to_analyze.replace('param_', '')} on {metric_to_analyze} was found.")
                        except Exception as e:
                            st.error(f"Error performing linear regression: {str(e)}")
                
            else:  # Box Plot
                fig = px.box(
                    results_df,
                    x=param_to_analyze,
                    y=metric_to_analyze,
                    title=f"Distribution of {metric_to_analyze} by {param_to_analyze.replace('param_', '')}",
                    labels={
                        param_to_analyze: param_to_analyze.replace("param_", ""),
                        metric_to_analyze: metric_to_analyze
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # ANOVA analysis for categorical variables
                groups = []
                for param_value in results_df[param_to_analyze].unique():
                    group = results_df[results_df[param_to_analyze] == param_value][metric_to_analyze].values
                    groups.append(group)
                
                if len(groups) >= 2 and all(len(g) > 0 for g in groups):
                    try:
                        f_val, p_val = stats.f_oneway(*groups)
                        st.write(f"ANOVA test: F-value: {f_val:.3f}, p-value: {p_val:.3f}")
                        
                        if p_val < 0.05:
                            st.success(f"There are significant differences in {metric_to_analyze} across different values of {param_to_analyze.replace('param_', '')}.")
                        else:
                            st.info(f"No significant differences in {metric_to_analyze} across different values of {param_to_analyze.replace('param_', '')}.")
                    except:
                        st.warning("Could not perform ANOVA test on this data.")
                
        else:  # Continuous parameter
            # Scatter plot with trend line
            fig = px.scatter(
                results_df,
                x=param_to_analyze,
                y=metric_to_analyze,
                trendline="ols",
                title=f"Relationship between {param_to_analyze.replace('param_', '')} and {metric_to_analyze}",
                labels={
                    param_to_analyze: param_to_analyze.replace("param_", ""),
                    metric_to_analyze: metric_to_analyze
                }
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation analysis
            corr = results_df[param_to_analyze].corr(results_df[metric_to_analyze])
            st.metric("Correlation Coefficient", f"{corr:.3f}")
            
            # Simple linear regression
            if len(results_df) > 1:
                x = results_df[param_to_analyze].values
                y = results_df[metric_to_analyze].values
                
                # Check if all x values are identical
                if np.std(x) == 0 or len(np.unique(x)) <= 1:
                    st.warning(f"Cannot perform linear regression: all {param_to_analyze.replace('param_', '')} values are identical.")
                else:
                    try:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                        
                        st.write(f"Linear regression: {metric_to_analyze} = {slope:.3f} × {param_to_analyze.replace('param_', '')} + {intercept:.3f}")
                        st.write(f"R²: {r_value**2:.3f}, p-value: {p_value:.3f}")
                        
                        # Interpretation
                        if p_value < 0.05:
                            if slope > 0:
                                st.success(f"There is a significant positive effect of {param_to_analyze.replace('param_', '')} on {metric_to_analyze}.")
                            else:
                                st.success(f"There is a significant negative effect of {param_to_analyze.replace('param_', '')} on {metric_to_analyze}.")
                        else:
                            st.info(f"No significant effect of {param_to_analyze.replace('param_', '')} on {metric_to_analyze} was found.")
                    except Exception as e:
                        st.error(f"Error performing linear regression: {str(e)}")
        
        # Correlation heatmap
        st.subheader("Correlation Analysis")
        
        # Select variables to include in correlation analysis
        corr_vars = st.multiselect(
            "Select Variables for Correlation Analysis",
            options=[col for col in results_df.columns if col not in ["database_id"]],
            default=[col for col in results_df.columns if col.startswith("param_") or col in ["avg_resonance", "final_relationship_score"]][:5]
        )
        
        if corr_vars and len(corr_vars) >= 2:
            corr_df = results_df[corr_vars].corr()
            
            fig = px.imshow(
                corr_df,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Correlation Matrix"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Highlight strongest correlations
            st.subheader("Key Correlations")
            
            # Get upper triangle of correlation matrix (excluding diagonal)
            corr_matrix = corr_df.values
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            corrs = corr_matrix[mask]
            var_pairs = [(corr_vars[i], corr_vars[j]) for i in range(len(corr_vars)) for j in range(i+1, len(corr_vars))]
            
            # Create dataframe of correlations
            corr_data = pd.DataFrame({
                "Variable 1": [pair[0] for pair in var_pairs],
                "Variable 2": [pair[1] for pair in var_pairs],
                "Correlation": corrs
            })
            
            # Sort by absolute correlation and display top 5
            top_corrs = corr_data.iloc[np.abs(corr_data["Correlation"]).argsort()[::-1]].head(5)
            
            for _, row in top_corrs.iterrows():
                var1 = row["Variable 1"].replace("param_", "") if row["Variable 1"].startswith("param_") else row["Variable 1"]
                var2 = row["Variable 2"].replace("param_", "") if row["Variable 2"].startswith("param_") else row["Variable 2"]
                corr = row["Correlation"]
                
                if abs(corr) > 0.7:
                    st.write(f"**Strong {'positive' if corr > 0 else 'negative'} correlation** ({corr:.2f}) between {var1} and {var2}")
                elif abs(corr) > 0.3:
                    st.write(f"**Moderate {'positive' if corr > 0 else 'negative'} correlation** ({corr:.2f}) between {var1} and {var2}")
                else:
                    st.write(f"**Weak {'positive' if corr > 0 else 'negative'} correlation** ({corr:.2f}) between {var1} and {var2}")
            
            # Interaction analysis
            st.subheader("Parameter Interaction Analysis")
            
            if len([col for col in results_df.columns if col.startswith("param_")]) >= 2:
                param1 = st.selectbox(
                    "First Parameter",
                    options=[col for col in results_df.columns if col.startswith("param_")],
                    format_func=lambda x: x.replace("param_", ""),
                    index=0,
                    key="param1"
                )
                
                param2 = st.selectbox(
                    "Second Parameter",
                    options=[col for col in results_df.columns if col.startswith("param_") and col != param1],
                    format_func=lambda x: x.replace("param_", ""),
                    index=0,
                    key="param2"
                )
                
                outcome = st.selectbox(
                    "Outcome Metric",
                    options=[col for col in results_df.columns if not col.startswith("param_") and col != "database_id"],
                    index=0,
                    key="outcome"
                )
                
                # Heatmap of interaction effect
                if results_df[param1].nunique() <= 10 and results_df[param2].nunique() <= 10:
                    pivot_table = pd.pivot_table(
                        results_df,
                        values=outcome,
                        index=param1,
                        columns=param2,
                        aggfunc="mean"
                    )
                    
                    fig = px.imshow(
                        pivot_table,
                        text_auto=".2f",
                        color_continuous_scale="Viridis",
                        title=f"Interaction Effect of {param1.replace('param_', '')} and {param2.replace('param_', '')} on {outcome}",
                        labels={
                            "x": param2.replace("param_", ""),
                            "y": param1.replace("param_", ""),
                            "color": outcome
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Interpretation
                    rows_diff = pivot_table.diff(axis=0).dropna().abs().mean().mean()
                    cols_diff = pivot_table.diff(axis=1).dropna().abs().mean().mean()
                    
                    if rows_diff > cols_diff * 1.5:
                        st.info(f"The effect of {param1.replace('param_', '')} on {outcome} appears stronger than the effect of {param2.replace('param_', '')}.")
                    elif cols_diff > rows_diff * 1.5:
                        st.info(f"The effect of {param2.replace('param_', '')} on {outcome} appears stronger than the effect of {param1.replace('param_', '')}.")
                    else:
                        st.info(f"Both {param1.replace('param_', '')} and {param2.replace('param_', '')} have similar levels of effect on {outcome}.")
                else:
                    st.info("Too many unique values for these parameters to create a meaningful heatmap.")
        else:
            st.info("Select at least two variables for correlation analysis.")
            
    else:
        st.info("No batch simulation results available. Run an experiment to see analysis here.")