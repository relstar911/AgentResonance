import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, entropy
import streamlit as st

def calculate_statistics(df):
    """Calculate summary statistics for the simulation"""
    stats = {
        'Relationship Score (Correlation)': pearsonr(df['A_B'], df['B_B'])[0] if len(df) > 1 else 0,
        'Average Resonance': df['resonance'].mean(),
        'Max Resonance': df['resonance'].max(),
        'Min Resonance': df['resonance'].min(),
        'Average A Bond Strength': df['A_bond'].mean(),
        'Average B Bond Strength': df['B_bond'].mean(),
        'Average Distance Between Agents': np.mean([
            np.sqrt((df['A_x'][i] - df['B_x'][i])**2 + (df['A_y'][i] - df['B_y'][i])**2) 
            for i in range(len(df))
        ]),
        'Proportion of Time Seeing Each Other': (df['A_sees_B'].sum() + df['B_sees_A'].sum()) / (2 * len(df))
    }
    
    return stats

def calculate_action_entropy(df):
    """Calculate entropy of agent actions"""
    # Calculate action distribution for agent A
    a_actions = df['A_action'].value_counts(normalize=True)
    b_actions = df['B_action'].value_counts(normalize=True)
    
    # Calculate entropy
    a_entropy = entropy(a_actions) if len(a_actions) > 0 else 0
    b_entropy = entropy(b_actions) if len(b_actions) > 0 else 0
    
    # Normalize entropy (max entropy for 5 actions is log(5))
    max_entropy = np.log(5)
    a_entropy_normalized = a_entropy / max_entropy if max_entropy > 0 else 0
    b_entropy_normalized = b_entropy / max_entropy if max_entropy > 0 else 0
    
    return {
        'Agent A Action Entropy': a_entropy,
        'Agent B Action Entropy': b_entropy,
        'Agent A Normalized Entropy': a_entropy_normalized,
        'Agent B Normalized Entropy': b_entropy_normalized,
        'A Action Distribution': a_actions.to_dict(),
        'B Action Distribution': b_actions.to_dict()
    }

def analyze_phi_delta_sigma(df):
    """Analyze trends in phi, delta, and sigma values"""
    analysis = {
        'A_phi_trend': np.polyfit(df['t'], df['A_phi'], 1)[0] if len(df) > 1 else 0,
        'B_phi_trend': np.polyfit(df['t'], df['B_phi'], 1)[0] if len(df) > 1 else 0,
        'A_delta_trend': np.polyfit(df['t'], df['A_delta'], 1)[0] if len(df) > 1 else 0,
        'B_delta_trend': np.polyfit(df['t'], df['B_delta'], 1)[0] if len(df) > 1 else 0,
        'A_sigma_trend': np.polyfit(df['t'], df['A_sigma'], 1)[0] if len(df) > 1 else 0,
        'B_sigma_trend': np.polyfit(df['t'], df['B_sigma'], 1)[0] if len(df) > 1 else 0,
        'A_phi_mean': df['A_phi'].mean(),
        'B_phi_mean': df['B_phi'].mean(),
        'A_delta_mean': df['A_delta'].mean(),
        'B_delta_mean': df['B_delta'].mean(),
        'A_sigma_mean': df['A_sigma'].mean(),
        'B_sigma_mean': df['B_sigma'].mean()
    }
    
    # Correlations between different metrics
    if len(df) > 1:
        analysis['correlation_A_phi_B_phi'] = pearsonr(df['A_phi'], df['B_phi'])[0]
        analysis['correlation_A_delta_B_delta'] = pearsonr(df['A_delta'], df['B_delta'])[0]
        analysis['correlation_A_sigma_B_sigma'] = pearsonr(df['A_sigma'], df['B_sigma'])[0]
    
    return analysis

def create_summary_table(stats):
    """Create a formatted summary table for statistics"""
    df_stats = pd.DataFrame({
        'Metric': list(stats.keys()),
        'Value': list(stats.values())
    })
    
    return df_stats

def display_statistics(stats):
    """Display statistics in a formatted way in Streamlit"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Relationship Score", f"{stats['Relationship Score (Correlation)']:.3f}")
        st.metric("Average Resonance", f"{stats['Average Resonance']:.3f}")
        st.metric("Average A Bond", f"{stats['Average A Bond Strength']:.3f}")
    
    with col2:
        st.metric("Max Resonance", f"{stats['Max Resonance']:.3f}")
        st.metric("Min Resonance", f"{stats['Min Resonance']:.3f}")
        st.metric("Average B Bond", f"{stats['Average B Bond Strength']:.3f}")
    
    st.metric("Average Distance Between Agents", f"{stats['Average Distance Between Agents']:.2f}")
    st.metric("Proportion of Time Seeing Each Other", f"{stats['Proportion of Time Seeing Each Other']:.2f}")
