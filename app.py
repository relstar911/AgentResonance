import streamlit as st
import pandas as pd
import numpy as np
import time
import base64
import matplotlib.pyplot as plt
import random
from io import BytesIO

# Import modules
import simulation
import visualization
import analysis

# Page configuration
st.set_page_config(
    page_title="Conscious Agent Simulation Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and description
st.title("Conscious Agent Simulation Dashboard")
st.markdown("""
This dashboard visualizes and analyzes simulations of Conscious Agents interacting in a grid world environment.
The agents exhibit behaviors influenced by their internal states (phi, sigma, delta, B) and form bonds with each other.
""")

# Sidebar for simulation configuration
st.sidebar.header("Simulation Parameters")

# Grid configuration
st.sidebar.subheader("Grid Configuration")
grid_size = st.sidebar.slider("Grid Size", min_value=5, max_value=20, value=10, step=1)
resources = st.sidebar.slider("Number of Resources", min_value=0, max_value=20, value=10, step=1)
dangers = st.sidebar.slider("Number of Dangers", min_value=0, max_value=10, value=5, step=1)
goals = st.sidebar.slider("Number of Goals", min_value=0, max_value=5, value=1, step=1)

# Agent configuration
st.sidebar.subheader("Agent Configuration")
custom_positions = st.sidebar.checkbox("Set Custom Starting Positions", value=False)

if custom_positions:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_a_x = st.number_input("Agent A X", min_value=0, max_value=grid_size-1, value=4)
        start_a_y = st.number_input("Agent A Y", min_value=0, max_value=grid_size-1, value=4)
    with col2:
        start_b_x = st.number_input("Agent B X", min_value=0, max_value=grid_size-1, value=6)
        start_b_y = st.number_input("Agent B Y", min_value=0, max_value=grid_size-1, value=6)
else:
    start_a_x, start_a_y = None, None
    start_b_x, start_b_y = None, None

# Simulation configuration
st.sidebar.subheader("Simulation Configuration")
steps = st.sidebar.slider("Number of Steps", min_value=10, max_value=500, value=100, step=10)
animation_speed = st.sidebar.slider("Animation Speed (ms)", min_value=50, max_value=1000, value=200, step=50)

# Run simulation button
run_button = st.sidebar.button("Run Simulation", type="primary")

# Initialize session state for storing simulation results
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'running_animation' not in st.session_state:
    st.session_state.running_animation = False

# Function to download data as CSV
def get_csv_download_link(df, filename="simulation_data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href

# Run simulation if button clicked
if run_button:
    with st.spinner("Running simulation..."):
        simulation_results = simulation.run_simulation(
            steps=steps,
            grid_size=grid_size,
            resources=resources,
            dangers=dangers,
            goals=goals,
            start_pos_a=(start_a_x, start_a_y),
            start_pos_b=(start_b_x, start_b_y)
        )
        st.session_state.simulation_results = simulation_results
        st.session_state.current_step = 0
        st.session_state.running_animation = False
        st.success("Simulation completed!")
        st.rerun()

# Display simulation results if available
if st.session_state.simulation_results:
    results = st.session_state.simulation_results
    grid = results['grid']
    log_df = results['log']
    agent_a = results['agent_a']
    agent_b = results['agent_b']
    relationship_score = results['relationship_score']
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Grid Visualization", "Metrics", "Path Analysis", "Statistics", "Raw Data"])
    
    with tab1:
        st.header("Grid World Visualization")
        
        # Control for animation
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            step_slider = st.slider("Step", min_value=0, max_value=steps-1, value=st.session_state.current_step)
            st.session_state.current_step = step_slider
        
        with col2:
            if st.button("Play/Pause Animation"):
                st.session_state.running_animation = not st.session_state.running_animation
        
        with col3:
            if st.button("Reset"):
                st.session_state.current_step = 0
                st.session_state.running_animation = False
                st.rerun()
        
        # Display the grid at the current step
        current_step = st.session_state.current_step
        agent_a_pos = (results['agent_positions'][current_step][0])
        agent_b_pos = (results['agent_positions'][current_step][1])
        
        # Display the grid
        visualization.display_grid(grid, agent_a_pos, agent_b_pos, step=current_step)
        
        # Show current metrics for this step
        step_data = log_df.iloc[current_step] if current_step < len(log_df) else log_df.iloc[-1]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Agent A")
            st.metric("Position", f"({step_data['A_x']}, {step_data['A_y']})")
            st.metric("Action", step_data['A_action'])
            st.metric("Sees B", "Yes" if step_data['A_sees_B'] else "No")
            st.metric("Bond Strength", f"{step_data['A_bond']:.3f}")
            
        with col2:
            st.subheader("Agent B")
            st.metric("Position", f"({step_data['B_x']}, {step_data['B_y']})")
            st.metric("Action", step_data['B_action'])
            st.metric("Sees A", "Yes" if step_data['B_sees_A'] else "No")
            st.metric("Bond Strength", f"{step_data['B_bond']:.3f}")
            
        with col3:
            st.subheader("Metrics")
            st.metric("A_Φ (phi)", f"{step_data['A_phi']:.3f}")
            st.metric("A_σ (sigma)", f"{step_data['A_sigma']:.3f}")
            st.metric("A_δ (delta)", f"{step_data['A_delta']:.3f}")
            st.metric("A_B", f"{step_data['A_B']:.3f}")
            
            st.metric("B_Φ (phi)", f"{step_data['B_phi']:.3f}")
            st.metric("B_σ (sigma)", f"{step_data['B_sigma']:.3f}")
            st.metric("B_δ (delta)", f"{step_data['B_delta']:.3f}")
            st.metric("B_B", f"{step_data['B_B']:.3f}")
            
            st.metric("Resonance", f"{step_data['resonance']:.3f}")
        
        # Animation logic
        if st.session_state.running_animation:
            if st.session_state.current_step < steps - 1:
                st.session_state.current_step += 1
                time.sleep(animation_speed / 1000)  # Convert ms to seconds
                st.rerun()
            else:
                st.session_state.running_animation = False
    
    with tab2:
        st.header("Agent Metrics Over Time")
        
        # B values and resonance plot
        st.subheader("B Values and Resonance")
        b_plot = visualization.plot_agent_b_values(log_df)
        st.plotly_chart(b_plot, use_container_width=True)
        
        # Bond strength plot
        st.subheader("Bond Strength")
        bond_plot = visualization.plot_bond_strength(log_df)
        st.plotly_chart(bond_plot, use_container_width=True)
        
        # Component metrics
        st.subheader("Component Metrics")
        
        metric_selector = st.selectbox(
            "Select Metrics to Display", 
            options=[
                "Agent A Components (phi, sigma, delta)",
                "Agent B Components (phi, sigma, delta)", 
                "Both Agents' phi",
                "Both Agents' sigma",
                "Both Agents' delta"
            ]
        )
        
        if metric_selector == "Agent A Components (phi, sigma, delta)":
            metrics = ["A_phi", "A_sigma", "A_delta"]
            title = "Agent A Component Metrics Over Time"
        elif metric_selector == "Agent B Components (phi, sigma, delta)":
            metrics = ["B_phi", "B_sigma", "B_delta"]
            title = "Agent B Component Metrics Over Time"
        elif metric_selector == "Both Agents' phi":
            metrics = ["A_phi", "B_phi"]
            title = "Phi Values for Both Agents"
        elif metric_selector == "Both Agents' sigma":
            metrics = ["A_sigma", "B_sigma"]
            title = "Sigma Values for Both Agents"
        elif metric_selector == "Both Agents' delta":
            metrics = ["A_delta", "B_delta"]
            title = "Delta Values for Both Agents"
        
        metrics_plot = visualization.plot_metrics_over_time(log_df, metrics, title)
        st.plotly_chart(metrics_plot, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        corr_plot = visualization.create_correlation_heatmap(log_df)
        st.plotly_chart(corr_plot, use_container_width=True)
    
    with tab3:
        st.header("Agent Path Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Agent A Path Heatmap")
            fig_a = visualization.create_agent_path_heatmap(agent_a, grid_size)
            st.pyplot(fig_a)
        
        with col2:
            st.subheader("Agent B Path Heatmap")
            fig_b = visualization.create_agent_path_heatmap(agent_b, grid_size)
            st.pyplot(fig_b)
        
        st.subheader("Agent Paths Visualization")
        path_plot = visualization.plot_agent_paths(log_df, grid_size)
        st.plotly_chart(path_plot, use_container_width=True)
        
        # Action analysis
        st.subheader("Agent Actions Analysis")
        action_stats = analysis.calculate_action_entropy(log_df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Agent A Action Entropy", f"{action_stats['Agent A Normalized Entropy']:.3f}")
            st.write("Action Distribution:")
            st.write(action_stats['A Action Distribution'])
        
        with col2:
            st.metric("Agent B Action Entropy", f"{action_stats['Agent B Normalized Entropy']:.3f}")
            st.write("Action Distribution:")
            st.write(action_stats['B Action Distribution'])
    
    with tab4:
        st.header("Statistical Analysis")
        
        # Calculate statistics
        stats = analysis.calculate_statistics(log_df)
        phi_delta_sigma_analysis = analysis.analyze_phi_delta_sigma(log_df)
        
        # Display statistics
        st.subheader("Summary Statistics")
        analysis.display_statistics(stats)
        
        st.subheader("Component Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Agent A Component Metrics:")
            st.metric("Average Phi", f"{phi_delta_sigma_analysis['A_phi_mean']:.3f}")
            st.metric("Average Sigma", f"{phi_delta_sigma_analysis['A_sigma_mean']:.3f}")
            st.metric("Average Delta", f"{phi_delta_sigma_analysis['A_delta_mean']:.3f}")
        
        with col2:
            st.write("Agent B Component Metrics:")
            st.metric("Average Phi", f"{phi_delta_sigma_analysis['B_phi_mean']:.3f}")
            st.metric("Average Sigma", f"{phi_delta_sigma_analysis['B_sigma_mean']:.3f}")
            st.metric("Average Delta", f"{phi_delta_sigma_analysis['B_delta_mean']:.3f}")
        
        # Display trend analysis
        st.subheader("Trend Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Agent A Trends:")
            st.metric("Phi Trend", f"{phi_delta_sigma_analysis['A_phi_trend']:.5f}", 
                      delta_color="normal")
            st.metric("Sigma Trend", f"{phi_delta_sigma_analysis['A_sigma_trend']:.5f}", 
                      delta_color="normal")
            st.metric("Delta Trend", f"{phi_delta_sigma_analysis['A_delta_trend']:.5f}", 
                      delta_color="normal")
        
        with col2:
            st.write("Agent B Trends:")
            st.metric("Phi Trend", f"{phi_delta_sigma_analysis['B_phi_trend']:.5f}", 
                      delta_color="normal")
            st.metric("Sigma Trend", f"{phi_delta_sigma_analysis['B_sigma_trend']:.5f}", 
                      delta_color="normal")
            st.metric("Delta Trend", f"{phi_delta_sigma_analysis['B_delta_trend']:.5f}", 
                      delta_color="normal")
        
        # Display correlation between agent metrics if available
        if len(log_df) > 1:
            st.subheader("Component Correlations Between Agents")
            
            st.metric("Phi Correlation", 
                      f"{phi_delta_sigma_analysis['correlation_A_phi_B_phi']:.3f}")
            st.metric("Sigma Correlation", 
                      f"{phi_delta_sigma_analysis['correlation_A_sigma_B_sigma']:.3f}")
            st.metric("Delta Correlation", 
                      f"{phi_delta_sigma_analysis['correlation_A_delta_B_delta']:.3f}")
    
    with tab5:
        st.header("Raw Simulation Data")
        
        # Show downloadable data
        st.subheader("Download Simulation Data")
        st.markdown(get_csv_download_link(log_df), unsafe_allow_html=True)
        
        # Display dataframe with filtered columns for better readability
        st.subheader("Data Preview")
        display_columns = st.multiselect(
            "Select columns to display",
            options=log_df.columns.tolist(),
            default=['t', 'A_x', 'A_y', 'B_x', 'B_y', 'A_B', 'B_B', 'resonance', 'A_bond', 'B_bond']
        )
        
        if display_columns:
            st.dataframe(log_df[display_columns])
        else:
            st.dataframe(log_df)

else:
    # If no simulation has been run yet
    st.info("Configure simulation parameters in the sidebar and click 'Run Simulation' to start.")
    
    # Show a sample grid to illustrate what will be generated
    st.subheader("Sample Grid")
    sample_grid = simulation.generate_grid(grid_size=grid_size, resources=resources, dangers=dangers, goals=goals)
    visualization.display_grid(
        sample_grid, 
        (random.randint(0, grid_size-1), random.randint(0, grid_size-1)),
        (random.randint(0, grid_size-1), random.randint(0, grid_size-1)),
        step=0
    )
    
    st.markdown("""
    ## About Conscious Agents
    
    The simulation models two conscious agents (A and B) moving in a grid world with:
    
    - **Resources** (green cells): Provide energy to agents
    - **Dangers** (red cells): Reduce agent energy
    - **Goals** (gold cells): Objectives for agents to reach
    
    ### Agent Properties
    
    Each agent has internal parameters that model aspects of consciousness:
    
    - **Φ (phi)**: Integration parameter measuring internal cohesion
    - **σ (sigma)**: Self-predictive accuracy
    - **δ (delta)**: Decision entropy/variety
    - **B**: Integrated information, calculated as phi × sigma × delta
    
    ### Relationship Dynamics
    
    Agents form bonds when they see each other, with bond strength affecting their movement decisions.
    Resonance between agents is measured by the similarity of their B values.
    
    ### Getting Started
    
    1. Adjust the simulation parameters in the sidebar
    2. Click "Run Simulation" to start
    3. Explore the different tabs to analyze the results
    """)

# Footer
st.markdown("---")
st.markdown("Conscious Agent Simulation Dashboard | Version 1.0")
