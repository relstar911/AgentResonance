import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

def generate_grid_colors(grid):
    """Convert numerical grid values to color names for visualization"""
    colors = np.empty(grid.shape, dtype=object)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == 0:
                colors[i, j] = 'lightgray'  # Empty cell
            elif grid[i, j] == 1:
                colors[i, j] = 'green'      # Resource
            elif grid[i, j] == -1:
                colors[i, j] = 'red'        # Danger
            elif grid[i, j] == 2:
                colors[i, j] = 'gold'       # Goal
    return colors

def display_grid(grid, agent_a_pos, agent_b_pos, step=0):
    """Display the grid world with agents at their positions"""
    colors = generate_grid_colors(grid)
    
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # Draw grid cells with appropriate colors
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color=colors[i, j], alpha=0.5))
    
    # Draw grid lines
    for i in range(grid.shape[0] + 1):
        ax.axhline(i, color='black', lw=1)
        ax.axvline(i, color='black', lw=1)
    
    # Place agents
    ax.plot(agent_a_pos[1] + 0.5, agent_a_pos[0] + 0.5, 'bo', markersize=15, label='Agent A')
    ax.plot(agent_b_pos[1] + 0.5, agent_b_pos[0] + 0.5, 'mo', markersize=15, label='Agent B')
    
    # Set plot limits and labels
    ax.set_aspect('equal')
    ax.set_xlim(0, grid.shape[1])
    ax.set_ylim(0, grid.shape[0])
    ax.invert_yaxis()  # Invert y-axis to match grid coordinates
    
    # Add legend and title
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    ax.set_title(f'Grid World - Step {step}')
    
    # Add explanation for colors
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color='lightgray', alpha=0.5, label='Empty'),
        plt.Rectangle((0, 0), 1, 1, color='green', alpha=0.5, label='Resource'),
        plt.Rectangle((0, 0), 1, 1, color='red', alpha=0.5, label='Danger'),
        plt.Rectangle((0, 0), 1, 1, color='gold', alpha=0.5, label='Goal')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    st.pyplot(fig)

def create_agent_path_heatmap(agent, grid_size):
    """Create a heatmap showing the agent's path frequency"""
    path_data = np.zeros((grid_size, grid_size))
    
    # Count occurrences of each position
    for pos in agent.path_history:
        path_data[pos] += 1
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(7, 7))
    sns.heatmap(path_data, cmap='Blues', ax=ax, cbar_kws={'label': 'Visit Frequency'})
    ax.set_title(f'Agent {agent.name} Path Heatmap')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    return fig

def plot_metrics_over_time(df, metrics, title):
    """Create a line chart showing metrics over time"""
    fig = px.line(df, x='t', y=metrics, title=title)
    fig.update_layout(
        xaxis_title='Simulation Step',
        yaxis_title='Value',
        legend_title='Metric'
    )
    return fig

def create_correlation_heatmap(df):
    """Create a correlation heatmap for agent metrics"""
    # Select relevant columns for correlation analysis
    cols = ['A_phi', 'A_sigma', 'A_delta', 'A_B', 'A_bond',
            'B_phi', 'B_sigma', 'B_delta', 'B_B', 'B_bond', 'resonance']
    
    corr = df[cols].corr()
    
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale=px.colors.sequential.Blues
    )
    fig.update_layout(
        title="Correlation Heatmap of Agent Metrics",
        height=700,
        width=700
    )
    
    return fig

def plot_agent_paths(df, grid_size):
    """Create an animated plot of agent paths"""
    # Create figure
    fig = go.Figure()
    
    # Add grid lines
    for i in range(grid_size + 1):
        fig.add_shape(
            type="line", x0=0, y0=i, x1=grid_size, y1=i,
            line=dict(color="gray", width=1)
        )
        fig.add_shape(
            type="line", x0=i, y0=0, x1=i, y1=grid_size,
            line=dict(color="gray", width=1)
        )
    
    # Add agent A path
    fig.add_trace(
        go.Scatter(
            x=df['A_y'], y=df['A_x'],
            mode='lines+markers',
            name='Agent A Path',
            line=dict(color='blue', width=2),
            marker=dict(size=8, symbol='circle')
        )
    )
    
    # Add agent B path
    fig.add_trace(
        go.Scatter(
            x=df['B_y'], y=df['B_x'],
            mode='lines+markers',
            name='Agent B Path',
            line=dict(color='magenta', width=2),
            marker=dict(size=8, symbol='circle')
        )
    )
    
    # Configure layout
    fig.update_layout(
        title="Agent Paths",
        xaxis=dict(
            title="Y Coordinate",
            range=[-0.5, grid_size - 0.5],
            dtick=1,
            zeroline=False
        ),
        yaxis=dict(
            title="X Coordinate",
            range=[-0.5, grid_size - 0.5],
            dtick=1,
            zeroline=False,
            autorange="reversed"  # Invert y-axis to match grid coordinates
        ),
        height=600,
        width=600,
        showlegend=True
    )
    
    return fig

def plot_agent_b_values(df):
    """Plot the B values of both agents over time"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add traces for both agents
    fig.add_trace(
        go.Scatter(x=df['t'], y=df['A_B'], name="Agent A B Value", line=dict(color='blue', width=2)),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=df['t'], y=df['B_B'], name="Agent B B Value", line=dict(color='magenta', width=2)),
        secondary_y=False
    )
    
    # Add resonance
    fig.add_trace(
        go.Scatter(x=df['t'], y=df['resonance'], name="Resonance", line=dict(color='green', width=2, dash='dot')),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title="B Values and Resonance Over Time",
        xaxis_title="Simulation Step",
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="B Value", secondary_y=False)
    fig.update_yaxes(title_text="Resonance", secondary_y=True)
    
    return fig

def plot_bond_strength(df):
    """Plot the bond strength of both agents over time"""
    fig = px.line(
        df, x='t', y=['A_bond', 'B_bond'], 
        title='Bond Strength Over Time',
        labels={'t': 'Simulation Step', 'value': 'Bond Strength', 'variable': 'Agent'}
    )
    
    fig.update_layout(
        xaxis_title='Simulation Step',
        yaxis_title='Bond Strength',
        legend_title='Agent',
        yaxis=dict(range=[0, 1])
    )
    
    return fig
