import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Import modules
import visualization
import analysis
import database

# Page configuration
st.set_page_config(
    page_title="Saved Simulations",
    page_icon="💾",
    layout="wide"
)

# Title and description
st.title("Saved Simulations")
st.markdown("""
Browse and analyze your saved simulation runs. Select a simulation from the list below to view its details.
""")

# Retrieve all saved simulations
simulations = database.get_all_simulations()

if not simulations:
    st.info("No saved simulations found. Run a simulation in the main app and save it to see it here.")
else:
    # Create a dataframe for easy display
    sim_data = []
    for sim in simulations:
        sim_data.append({
            "ID": sim.id,
            "Name": sim.name if sim.name else f"Simulation {sim.id}",
            "Date": sim.created_at.strftime("%Y-%m-%d %H:%M"),
            "Grid Size": sim.grid_size,
            "Steps": sim.steps,
            "Relationship Score": round(sim.relationship_score, 3) if sim.relationship_score else None,
            "Avg Resonance": round(sim.avg_resonance, 3) if sim.avg_resonance else None
        })
    
    sim_df = pd.DataFrame(sim_data)
    
    # Display simulations as a table
    st.dataframe(sim_df, use_container_width=True)
    
    # Select a simulation to view
    selected_sim_id = st.selectbox(
        "Select a simulation to view details:",
        options=[sim.id for sim in simulations],
        format_func=lambda x: next((s["Name"] for s in sim_data if s["ID"] == x), f"Simulation {x}")
    )
    
    if st.button("Load Simulation"):
        # Fetch the selected simulation data
        sim_data = database.get_simulation(selected_sim_id)
        
        if sim_data:
            display_name = sim_data['name'] if sim_data['name'] else f"Simulation {sim_data['id']}"
            st.success(f"Loaded simulation: {display_name}")
            
            # Create tabs for different visualizations
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Grid Visualization", "Metrics", "Path Analysis", "Raw Data"])
            
            with tab1:
                st.header("Simulation Overview")
                
                # Display basic metadata
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Simulation Details")
                    st.metric("ID", sim_data['id'])
                    st.metric("Name", sim_data['name'] if sim_data['name'] else "Unnamed")
                    st.metric("Date", sim_data['created_at'].strftime("%Y-%m-%d %H:%M"))
                    st.metric("Grid Size", sim_data['grid_size'])
                    st.metric("Steps", sim_data['steps'])
                
                with col2:
                    st.subheader("Results Summary")
                    st.metric("Relationship Score", f"{sim_data['relationship_score']:.3f}")
                    st.metric("Average Resonance", f"{sim_data['avg_resonance']:.3f}")
                    st.metric("Agent A Avg Bond", f"{sim_data['avg_bond_strength_a']:.3f}")
                    st.metric("Agent B Avg Bond", f"{sim_data['avg_bond_strength_b']:.3f}")
                
                # Display the description if available
                if sim_data['description']:
                    st.subheader("Description")
                    st.markdown(sim_data['description'])
                
                # Delete simulation option
                if st.button("Delete Simulation", type="primary"):
                    if database.delete_simulation(selected_sim_id):
                        st.success(f"Simulation {selected_sim_id} deleted!")
                        st.rerun()
                    else:
                        st.error("Failed to delete simulation.")
            
            with tab2:
                st.header("Grid World Visualization")
                
                # Load grid data
                grid = np.array(sim_data['grid'])
                
                # Set up slider for step selection
                step = st.slider("Step", 0, sim_data['steps']-1, 0)
                
                # Get agent positions for the selected step
                agent_positions = sim_data['agent_positions']
                if step < len(agent_positions):
                    agent_a_pos = tuple(agent_positions[step][0])
                    agent_b_pos = tuple(agent_positions[step][1])
                    
                    # Display grid
                    visualization.display_grid(grid, agent_a_pos, agent_b_pos, step=step)
                    
                    # Get log data for this step
                    log_df = sim_data['log']
                    step_data = log_df.iloc[step] if step < len(log_df) else log_df.iloc[-1]
                    
                    # Show metrics for this step
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
            
            with tab3:
                st.header("Agent Metrics Over Time")
                
                log_df = sim_data['log']
                
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
            
            with tab4:
                st.header("Agent Path Analysis")
                
                log_df = sim_data['log']
                
                # Display agent paths visualization
                st.subheader("Agent Paths Visualization")
                path_plot = visualization.plot_agent_paths(log_df, sim_data['grid_size'])
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
            
            with tab5:
                st.header("Raw Simulation Data")
                
                log_df = sim_data['log']
                
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