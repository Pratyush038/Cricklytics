import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os

# Set page config
st.set_page_config(layout="wide", page_title="Bowler Injury Prediction Dashboard")

# Title and description
st.title("Bowler Injury Prediction Analysis")
st.markdown("This dashboard visualizes data for three bowlers and displays injury prediction metrics.")

# File uploader for player data
st.sidebar.header("Upload Player Data")
player1_file = st.sidebar.file_uploader("Upload Player 1 Data (CSV)", type=['csv'])
player2_file = st.sidebar.file_uploader("Upload Player 2 Data (CSV)", type=['csv'])
player3_file = st.sidebar.file_uploader("Upload Player 3 Data (CSV)", type=['csv'])

# File uploader for model
model_file = st.sidebar.file_uploader("Upload Trained Model (joblib)", type=['joblib', 'pkl'])

# Load data function
def load_player_data(file, player_name):
    if file is not None:
        df = pd.read_csv(file)
        df['Player'] = player_name
        return df
    return None

# Load model function
def load_model(file):
    if file is not None:
        model = joblib.load(file)
        return model
    return None

# Process data
player1_data = load_player_data(player1_file, "Player 1")
player2_data = load_player_data(player2_file, "Player 2")
player3_data = load_player_data(player3_file, "Player 3")
model = load_model(model_file)

# Combine available data
player_dfs = [df for df in [player1_data, player2_data, player3_data] if df is not None]
if player_dfs:
    all_data = pd.concat(player_dfs, ignore_index=True)
    
    # Data is loaded, show controls
    st.sidebar.header("Dashboard Controls")
    
    available_players = all_data['Player'].unique().tolist()
    selected_players = st.sidebar.multiselect(
        "Select Players to Display",
        options=available_players,
        default=available_players
    )
    
    # Dynamically get available metrics from data
    available_metrics = [col for col in all_data.columns if col not in ['Player', 'index']]
    metrics_to_view = st.sidebar.multiselect(
        "Select Metrics to Display",
        options=available_metrics,
        default=available_metrics[:2] if len(available_metrics) > 1 else available_metrics
    )
    
    visualization_type = st.sidebar.selectbox(
        "Select Visualization Type",
        options=['Line Charts', 'Bar Charts', 'Radar Charts', 'Heatmaps', 'Combined View']
    )
    
    # Filter data based on selection
    filtered_data = all_data[all_data['Player'].isin(selected_players)]
    
    # Determine if we have time-series data
    time_column = None
    for col in filtered_data.columns:
        if any(keyword in col.lower() for keyword in ['session', 'date', 'time', 'day', 'week']):
            time_column = col
            break
    
    if time_column is None and 'index' in filtered_data.columns:
        time_column = 'index'
    
    # Main content
    st.header("Data Visualization")
    
    def display_line_charts():
        for metric in metrics_to_view:
            if metric in filtered_data.columns and metric != time_column:
                if time_column:
                    fig = px.line(filtered_data, x=time_column, y=metric, color='Player', 
                                 title=f"{metric} Over Time by Player", 
                                 markers=True)
                else:
                    # If no time column, use index
                    fig = px.line(filtered_data, y=metric, color='Player', 
                                 title=f"{metric} by Player", 
                                 markers=True)
                st.plotly_chart(fig, use_container_width=True)
    
    def display_bar_charts():
        for metric in metrics_to_view:
            if metric in filtered_data.columns and metric != time_column:
                if time_column:
                    fig = px.bar(filtered_data, x=time_column, y=metric, color='Player', 
                                title=f"{metric} Comparison by Player",
                                barmode='group')
                else:
                    # Group by player if no time column
                    player_means = filtered_data.groupby('Player')[metric].mean().reset_index()
                    fig = px.bar(player_means, x='Player', y=metric,
                                title=f"Average {metric} by Player")
                st.plotly_chart(fig, use_container_width=True)
    
    def display_radar_charts():
        # Create radar chart for each player
        for player in selected_players:
            player_data = filtered_data[filtered_data['Player'] == player]
            
            # Calculate average of each metric for radar chart
            avg_metrics = {}
            for metric in metrics_to_view:
                if metric in player_data.columns and metric != time_column:
                    avg_metrics[metric] = player_data[metric].mean()
            
            if avg_metrics:
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=list(avg_metrics.values()),
                    theta=list(avg_metrics.keys()),
                    fill='toself',
                    name=player
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True
                        ),
                    ),
                    showlegend=True,
                    title=f"Radar Chart for {player}"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def display_heatmaps():
        # Create correlation heatmap for each player
        for player in selected_players:
            player_data = filtered_data[filtered_data['Player'] == player]
            
            # Get numeric columns
            numeric_cols = player_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if time_column in numeric_cols:
                numeric_cols.remove(time_column)
            
            if len(numeric_cols) > 1:  # Need at least 2 columns for correlation
                corr = player_data[numeric_cols].corr()
                
                fig = px.imshow(corr, 
                              text_auto=True, 
                              title=f"Correlation Heatmap for {player}",
                              color_continuous_scale='RdBu_r',
                              zmin=-1, zmax=1)
                st.plotly_chart(fig, use_container_width=True)
    
    def display_combined_view():
        # Create side-by-side comparison of key metrics
        if time_column:
            # For time series data, show line chart with all players
            for metric in metrics_to_view[:2]:  # Limit to first 2 metrics to avoid clutter
                if metric in filtered_data.columns and metric != time_column:
                    fig = px.line(filtered_data, x=time_column, y=metric, color='Player',
                                 title=f"{metric} Comparison",
                                 markers=True)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Show box plots for distribution comparison
        for metric in metrics_to_view[:3]:  # Limit to first 3 metrics
            if metric in filtered_data.columns and metric != time_column:
                fig = px.box(filtered_data, x='Player', y=metric, 
                           title=f"{metric} Distribution by Player")
                st.plotly_chart(fig, use_container_width=True)
    
    # Display visualizations based on selection
    if visualization_type == 'Line Charts':
        display_line_charts()
    elif visualization_type == 'Bar Charts':
        display_bar_charts()
    elif visualization_type == 'Radar Charts':
        display_radar_charts()
    elif visualization_type == 'Heatmaps':
        display_heatmaps()
    elif visualization_type == 'Combined View':
        display_combined_view()
    
    # Show data table
    st.header("Raw Data")
    st.dataframe(filtered_data)
    
    # Make predictions if model is loaded
    if model is not None:
        st.header("Injury Predictions")
        
        try:
            # Get feature columns from the model
            # This assumes the model has a 'feature_names_in_' attribute (like scikit-learn models)
            if hasattr(model, 'feature_names_in_'):
                feature_cols = model.feature_names_in_
            else:
                # Fallback - use all numeric columns except identified non-features
                non_features = ['Player', time_column] if time_column else ['Player']
                feature_cols = [col for col in filtered_data.select_dtypes(include=['float64', 'int64']).columns 
                               if col not in non_features]
            
            # Check if all required features are present
            missing_features = [col for col in feature_cols if col not in filtered_data.columns]
            
            if not missing_features:
                # Make predictions for each player
                for player in selected_players:
                    player_data = filtered_data[filtered_data['Player'] == player]
                    
                    if len(player_data) > 0:
                        X = player_data[feature_cols]
                        
                        # Get predictions
                        try:
                            if hasattr(model, 'predict_proba'):
                                predictions = model.predict_proba(X)[:, 1]  # Probability of positive class
                                pred_label = "Injury Risk Probability"
                            else:
                                predictions = model.predict(X)
                                pred_label = "Predicted Value"
                            
                            # Add predictions to data
                            player_pred_data = player_data.copy()
                            player_pred_data[pred_label] = predictions
                            
                            # Display prediction chart
                            if time_column:
                                fig = px.line(player_pred_data, x=time_column, y=pred_label,
                                            title=f"{pred_label} for {player}",
                                            markers=True)
                            else:
                                fig = px.line(player_pred_data, y=pred_label,
                                            title=f"{pred_label} for {player}",
                                            markers=True)
                            
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error making predictions for {player}: {e}")
                
            else:
                st.error(f"Missing features required by the model: {', '.join(missing_features)}")
        
        except Exception as e:
            st.error(f"Error using the model: {e}")
else:
    st.info("Please upload data files for at least one player to begin visualization.")