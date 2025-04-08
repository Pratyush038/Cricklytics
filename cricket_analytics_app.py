import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import time

st.set_page_config(
    page_title="Cricklytics",
    page_icon="🏏",
    layout="wide"
)

# Custom CSS with improved container styling and animations
st.markdown("""
<style>
    /* Main styling and typography */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        animation: fadeIn 1s ease-in-out;
    }
    
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2563EB;
        margin-bottom: 0.8rem;
        padding-top: 0.5rem;
    }
    
    /* Card styling with proper padding and overflow control */
    .card {
        background-color: #F3F4F6;
        border-radius: 0.5rem;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        overflow: hidden;
        word-wrap: break-word;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    .card-content {
        padding: 0.8rem;
        overflow: hidden;
        word-wrap: break-word;
    }
    
    /* Report sections styling */
    .report-header {
        background-color: #1E3A8A;
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 0.5rem 0.5rem 0 0;
        font-weight: 600;
        font-size: 1.2rem;
        transition: background-color 0.3s ease;
    }
    
    .report-header:hover {
        background-color: #2563EB;
    }
    
    .report-body {
        background-color: white;
        padding: 1.8rem;
        border-radius: 0 0 0.5rem 0.5rem;
        border: 1px solid #E5E7EB;
        overflow: hidden;
        word-wrap: break-word;
        margin-bottom: 2rem;
    }
    
    /* Model info card */
    .model-info {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
        border-radius: 0.25rem;
        transition: background-color 0.3s ease;
    }
    
    .model-info:hover {
        background-color: #DBEAFE;
    }
    
    /* List styling for recommendations */
    .recommendation-item {
        padding: 0.5rem 0;
        border-bottom: 1px solid #F3F4F6;
        transition: background-color 0.2s ease;
    }
    
    .recommendation-item:hover {
        background-color: #F9FAFB;
    }
    
    /* Button styling */
    .stButton > button {
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Animation keyframes */
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    /* Fix for tab content */
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1rem;
        overflow: auto;
    }
    
    /* Fix for text overflow in all containers */
    div[data-testid="stVerticalBlock"] {
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    
    /* Streamlit element fixes */
    .element-container {
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    
    /* Ensure text in columns doesn't overflow */
    .row-widget.stHorizontal {
        flex-wrap: wrap;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 1rem;
        color: #6B7280;
        font-size: 0.875rem;
        border-top: 1px solid #E5E7EB;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# App Header with animation effect
def display_header():
    st.markdown("<h1 class='main-header'>🏏 Cricklytics</h1>", unsafe_allow_html=True)
    
    # Add a subheading with animation
    st.markdown(
        """
        <div style="text-align: center; animation: fadeIn 1.5s ease-in-out; margin-bottom: 1.5rem;">
            <p style="font-size: 1.2rem; color: #4B5563;">
                Analyze cricket player performance with AI-powered insights from pre-trained models
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )

display_header()

# Hardcoded model functionality
def classify_batter(player_data):
    """
    Classify a new batter based on the hardcoded model logic
    
    Parameters:
    player_data (dict): Player statistics
    
    Returns:
    str: Predicted role
    """
    # Extract player stats
    mat = player_data.get('mat', 0)
    runs = player_data.get('runs', 0)
    avg = player_data.get('avg', 0)
    sr = player_data.get('sr', 0)
    career_length = player_data.get('career_length', 0)
    
    # Calculate boundary percentage if provided
    if 'fours' in player_data and 'sixes' in player_data:
        fours = player_data.get('fours', 0)
        sixes = player_data.get('sixes', 0)
        boundary_pct = ((4 * fours + 6 * sixes) / runs) * 100 if runs > 0 else 0
    else:
        boundary_pct = player_data.get('boundary_pct', 0)
    
    # Define thresholds (simplified from your script)
    anchor_avg_threshold = 35.0
    power_sr_threshold = 130.0
    power_boundary_threshold = 45.0
    finisher_sr_threshold = 140.0
    experience_threshold_years = 4.5
    experience_threshold_matches = 20
    experience_threshold_runs = 500
    
    # Apply binary classifications
    is_anchor = (avg >= anchor_avg_threshold) and (runs > 1000)
    is_power_hitter = (sr >= power_sr_threshold) and (boundary_pct >= power_boundary_threshold)
    is_finisher = (sr >= finisher_sr_threshold) and (mat > 10)
    is_experienced = (career_length > experience_threshold_years) and \
                    (mat >= experience_threshold_matches) and \
                    (runs >= experience_threshold_runs)
    
    # Determine composite role
    experience = "Experienced" if is_experienced else "Inexperienced"
    
    if is_anchor and is_power_hitter and is_finisher:
        style = "Elite Batter"
    elif is_anchor and is_power_hitter:
        style = "Versatile Batter"
    elif is_anchor and is_finisher:
        style = "Anchor Finisher"
    elif is_power_hitter and is_finisher:
        style = "Explosive Finisher"
    elif is_anchor:
        style = "Anchor"
    elif is_power_hitter:
        style = "Power Hitter"
    elif is_finisher:
        style = "Finisher"
    else:
        style = "Balanced Batter"
    
    return f"{experience} {style}"

def classify_bowler(player_data):
    """
    Classify a new bowler based on the hardcoded model logic
    
    Parameters:
    player_data (dict): Player statistics
    
    Returns:
    str: Predicted role
    """
    # Extract player stats
    mat = player_data.get('mat', 0)
    wickets = player_data.get('wickets', 0)
    econ = player_data.get('econ', 0)
    sr = player_data.get('sr', 0)
    career_length = player_data.get('career_length', 0)
    
    # Define thresholds (simplified from your script)
    wicket_taker_threshold = 25.0  # Lower strike rate is better
    economist_threshold = 6.5      # Lower economy is better
    experience_threshold_years = 4.5
    experience_threshold_matches = 50
    
    # Apply binary classifications
    is_wicket_taker = (sr <= wicket_taker_threshold) and (wickets > 50)
    is_economist = econ <= economist_threshold
    is_experienced = (career_length > experience_threshold_years) and (mat > experience_threshold_matches)
    
    # Determine composite role
    experience = "Experienced" if is_experienced else "Inexperienced"
    
    if is_wicket_taker and is_economist:
        style = "Elite Bowler"
    elif is_wicket_taker:
        style = "Wicket Taker"
    elif is_economist:
        style = "Economist"
    else:
        style = "Balanced Bowler"
    
    return f"{experience} {style}"

# Sidebar for model selection with improved styling
with st.sidebar:
    st.markdown("<h2 class='sub-header'>Player Category</h2>", unsafe_allow_html=True)
    
    model_category = st.radio("Select Player Category", ["Batsman", "Bowler"])
    
    st.markdown("<div class='model-info'>", unsafe_allow_html=True)
    st.markdown("#### Pre-trained Model Info")
    if model_category == "Batsman":
        st.write("Using integrated batting analysis model")
        st.write("- Features: matches, runs, strike rate, average, boundary %, career span")
        st.write("- Categories: Elite, Anchor, Power Hitter, Finisher, etc.")
    else:
        st.write("Using integrated bowling analysis model")
        st.write("- Features: matches, wickets, economy, strike rate, career span")
        st.write("- Categories: Elite, Wicket Taker, Economist, etc.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.divider()
    st.markdown("### About")
    st.write("This app uses machine learning to analyze cricket player stats and generate performance insights.")

# Custom tab styling with animation
tab_style = """
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        transition: background-color 0.3s ease, color 0.3s ease;
        border-radius: 4px 4px 0 0;
        padding: 8px 16px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1E3A8A !important;
        color: white !important;
    }
</style>
"""
st.markdown(tab_style, unsafe_allow_html=True)

# Main app content with improved tabs
tab1, tab2 = st.tabs(["Player Analysis", "Results"])

with tab1:
    st.markdown("<h2 class='sub-header'>Player Statistics Input</h2>", unsafe_allow_html=True)
    
    if model_category == "Batsman":
        col1, col2 = st.columns(2)
        
        with col1:
            with st.container():
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<div class='card-content'>", unsafe_allow_html=True)
                player_name = st.text_input("Player Name")
                matches = st.number_input("Matches Played", min_value=1, value=50)
                runs = st.number_input("Total Runs", min_value=0, value=1500)
                strike_rate = st.number_input("Strike Rate", min_value=0.0, value=85.0, format="%.2f")
                st.markdown("</div></div>", unsafe_allow_html=True)
        
        with col2:
            with st.container():
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<div class='card-content'>", unsafe_allow_html=True)
                average = st.number_input("Batting Average", min_value=0.0, value=35.0, format="%.2f")
                boundary_percent = st.slider("Boundary Percentage", 0.0, 100.0, 40.0, format="%.1f")
                start_year = st.number_input("Career Start Year", min_value=1950, max_value=datetime.now().year, value=2010)
                end_year = st.number_input("Career End Year (Current year if active)", min_value=1950, max_value=datetime.now().year, value=datetime.now().year)
                st.markdown("</div></div>", unsafe_allow_html=True)
            
        career_length = end_year - start_year + 1
            
    elif model_category == "Bowler":
        col1, col2 = st.columns(2)
        
        with col1:
            with st.container():
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<div class='card-content'>", unsafe_allow_html=True)
                player_name = st.text_input("Player Name")
                matches = st.number_input("Matches Played", min_value=1, value=50)
                wickets = st.number_input("Total Wickets", min_value=0, value=75)
                st.markdown("</div></div>", unsafe_allow_html=True)
        
        with col2:
            with st.container():
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<div class='card-content'>", unsafe_allow_html=True)
                economy = st.number_input("Economy Rate", min_value=0.0, value=4.5, format="%.2f")
                strike_rate = st.number_input("Bowling Strike Rate", min_value=0.0, value=30.0, format="%.2f")
                start_year = st.number_input("Career Start Year", min_value=1950, max_value=datetime.now().year, value=2010)
                end_year = st.number_input("Career End Year (Current year if active)", min_value=1950, max_value=datetime.now().year, value=datetime.now().year)
                st.markdown("</div></div>", unsafe_allow_html=True)
            
        career_length = end_year - start_year + 1
    
    # Add a visual separator before the button
    st.markdown("""
    <div style="height: 1px; background-color: #E5E7EB; margin: 1rem 0;"></div>
    """, unsafe_allow_html=True)
    
    # Improved button with animation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("Generate Analysis Report", type="primary", use_container_width=True)

# Function to generate analysis based on player data
def generate_analysis(player_data, player_type):
    try:
        # Add a small delay for animation effect
        with st.spinner("Analyzing player data..."):
            time.sleep(0.8)
            
        # Classification based on model type
        if player_type == "Batsman":
            prediction = classify_batter(player_data)
            
            # Generate insights based on stats and prediction
            insights = []
            
            # Strike rate insights
            if player_data["sr"] > 150:
                insights.append("Exceptional strike rate indicates an aggressive batting approach.")
            elif player_data["sr"] > 135:
                insights.append("Good strike rate shows ability to score quickly when needed.")
            elif player_data["sr"] > 120:
                insights.append("Moderate strike rate suggests a balanced approach to batting.")
            else:
                insights.append("Conservative strike rate indicates focus on building innings rather than quick scoring.")
                
            # Average insights
            if player_data["avg"] > 38:
                insights.append("Excellent batting average demonstrates consistent high performance.")
            elif player_data["avg"] > 30:
                insights.append("Good batting average shows reliability at the crease.")
            elif player_data["avg"] > 22:
                insights.append("Moderate batting average indicates room for improvement in consistency.")
            else:
                insights.append("Lower batting average suggests need for technical refinement.")
                
            # Boundary percentage insights
            if player_data["boundary_pct"] > 60:
                insights.append("Very high boundary percentage shows heavy reliance on boundaries for scoring.")
            elif player_data["boundary_pct"] > 45:
                insights.append("Good boundary percentage indicates strong ability to find gaps and hit boundaries.")
            elif player_data["boundary_pct"] > 30:
                insights.append("Moderate boundary percentage suggests balanced scoring between boundaries and running.")
            else:
                insights.append("Lower boundary percentage indicates preference for rotating strike over boundary hitting.")
                
            # Experience insights
            if player_data["career_length"] > 10:
                insights.append("Long career span demonstrates durability and sustained performance at high level.")
            elif player_data["career_length"] > 5:
                insights.append("Substantial career span shows established presence in competitive cricket.")
            else:
                insights.append("Shorter career span indicates a developing player with potential for growth.")
                
            # Match insights
            if player_data["mat"] > 100:
                insights.append("High number of matches played shows extensive experience in competitive cricket.")
            elif player_data["mat"] > 50:
                insights.append("Good number of matches indicates solid experience level.")
            else:
                insights.append("Lower match count suggests emerging talent with room to grow.")
            
        else:  # Bowler
            prediction = classify_bowler(player_data)
            
            # Generate insights based on stats and prediction
            insights = []
            
            # Economy insights
            if player_data["econ"] < 6.0:
                insights.append("Exceptional economy rate demonstrates elite containment ability.")
            elif player_data["econ"] < 7.0:
                insights.append("Very good economy rate shows strong control over runs conceded.")
            elif player_data["econ"] < 8.0:
                insights.append("Reasonable economy rate indicates decent control in limiting run scoring.")
            else:
                insights.append("Higher economy rate suggests need for improved control and containment.")
                
            # Strike rate insights for bowlers (lower is better)
            if player_data["sr"] < 25:
                insights.append("Outstanding strike rate shows excellent wicket-taking ability.")
            elif player_data["sr"] < 30:
                insights.append("Very good strike rate demonstrates strong capability to take wickets regularly.")
            elif player_data["sr"] < 35:
                insights.append("Decent strike rate indicates ability to breakthrough oppositions.")
            else:
                insights.append("Higher strike rate suggests room for improvement in wicket-taking effectiveness.")
                
            # Wickets insights
            if player_data["wickets"] > 150:
                insights.append("Impressive wicket tally demonstrates consistent success over career.")
            elif player_data["wickets"] > 75:
                insights.append("Good wicket haul shows proven effectiveness against quality opposition.")
            elif player_data["wickets"] > 25:
                insights.append("Growing wicket count indicates developing proficiency.")
            else:
                insights.append("Lower wicket count suggests early career stage with potential to improve.")
                
            # Experience insights
            if player_data["career_length"] > 10:
                insights.append("Long career span demonstrates durability and consistent performance.")
            elif player_data["career_length"] > 5:
                insights.append("Substantial career indicates established presence in competitive cricket.")
            else:
                insights.append("Shorter career span suggests a developing bowler with growth potential.")
        
        return {
            "prediction": prediction,
            "insights": insights
        }
        
    except Exception as e:
        st.error(f"Error generating analysis: {str(e)}")
        # Log the full exception for debugging
        import traceback
        st.write(traceback.format_exc())
        return {
            "prediction": "Error in analysis",
            "insights": [f"An error occurred: {str(e)}"]
        }

# Function to create visualization for player stats with improved styling
def create_player_visualization(player_data, player_type):
    # Set a cohesive visual style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#F9FAFB')
    
    # Define a custom color palette
    colors = ['#2563EB', '#1E40AF', '#3B82F6', '#60A5FA', '#93C5FD']
    
    if player_type == "Batsman":
        # Radar chart data
        categories = ['Matches', 'Runs', 'Strike Rate', 'Average', 'Boundary %']
        
        # Normalized values (0-1 scale)
        values = [
            min(1.0, player_data["mat"] / 200),
            min(1.0, player_data["runs"] / 10000),
            min(1.0, player_data["sr"] / 150),
            min(1.0, player_data["avg"] / 60),
            player_data["boundary_pct"] / 100
        ]
        # Create a polar subplot (this is the fix)
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        # Create the radar chart
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # Close the polygon
        angles += angles[:1]  # Close the polygon
        
        ax.plot(angles, values, linewidth=2.5, linestyle='solid', color=colors[0])
        ax.fill(angles, values, alpha=0.4, color=colors[2])
        
        # Improve radar chart styling
        ax.set_xticks(angles[:-1])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
            label.set_fontsize(9)
            label.set_fontweight('semibold')
            label.set_horizontalalignment('center')
            label.set_verticalalignment('center')
            x, y = np.cos(angle), np.sin(angle)
            label.set_position((1.12 * x, 1.12 * y))

        ax.set_title(f"Batting Performance - {player_data['name']}", fontsize=14, fontweight='bold', pad=20)
        
        # Add grid lines
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8], angle=0, fontsize=8)
        
        # Add subtle annotations for context
        for i, v in enumerate(values[:-1]):
            ax.text(angles[i], v + 0.05, f"{int(v*100)}%", 
                   color=colors[0], fontweight='bold', ha='center', va='center')
        
    elif player_type == "Bowler":
        # Radar chart data
        categories = ['Matches', 'Wickets', 'Economy', 'Strike Rate']
        
        # For economy and strike rate, lower is better, so we invert the scale
        economy_normalized = 1 - min(1.0, player_data["econ"] / 10)
        strike_rate_normalized = 1 - min(1.0, player_data["sr"] / 100)
        
        # Normalized values (0-1 scale)
        values = [
            min(1.0, player_data["mat"] / 200),
            min(1.0, player_data["wickets"] / 300),
            economy_normalized,
            strike_rate_normalized
        ]
        
        # Create the radar chart
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # Close the polygon
        angles += angles[:1]  # Close the polygon
        
        ax.plot(angles, values, linewidth=2.5, linestyle='solid', color=colors[1])
        ax.fill(angles, values, alpha=0.4, color=colors[3])
        
        # Improve radar chart styling
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
        ax.set_title(f"Bowling Performance - {player_data['name']}", fontsize=14, fontweight='bold', pad=20)
        
        # Add grid lines
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8], angle=0, fontsize=8)
        
        # Add subtle annotations for context
        for i, v in enumerate(values[:-1]):
            ax.text(angles[i], v + 0.05, f"{int(v*100)}%", 
                   color=colors[1], fontweight='bold', ha='center', va='center')
    
    plt.tight_layout()
    return fig

# Generate player comparison chart with improved styling
def create_player_comparison(player_data, player_type):
    # Set a cohesive visual style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#F9FAFB')
    
    # Define a consistent color palette
    colors = ['#2563EB', '#1E40AF', '#3B82F6', '#60A5FA', '#93C5FD']
    
    if player_type == "Batsman":
        # Sample reference values for different player types
        reference_data = {
            "Anchor": {"sr": 70, "avg": 45, "boundary_pct": 30},
            "Power Hitter": {"sr": 140, "avg": 30, "boundary_pct": 60},
            "Balanced": {"sr": 100, "avg": 35, "boundary_pct": 45},
            "Elite": {"sr": 130, "avg": 50, "boundary_pct": 55},
            "Current Player": {"sr": player_data["sr"], "avg": player_data["avg"], "boundary_pct": player_data["boundary_pct"]}
        }
        
        # Metrics to compare
        metrics = ["sr", "avg", "boundary_pct"]
        metric_labels = ["Strike Rate", "Average", "Boundary %"]
        
        # Set x positions and width
        x = np.arange(len(metrics))
        width = 0.15
        
        # Plot bars with improved styling
        for i, (role, values) in enumerate(reference_data.items()):
            offset = width * (i - len(reference_data)/2 + 0.5)
            values_list = [values[m] for m in metrics]
            rects = ax.bar(x + offset, values_list, width, label=role, color=colors[i % len(colors)], 
                          alpha=0.8 if role != "Current Player" else 1.0,
                          edgecolor='white', linewidth=0.7)
            
            # Add value labels on top of bars
            if role == "Current Player":
                for j, v in enumerate(values_list):
                    ax.text(x[j] + offset, v + 2, f'{v:.1f}', ha='center', va='bottom', 
                           fontsize=8, fontweight='bold', color=colors[i % len(colors)])
        
        # Customize plot
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold', labelpad=10)
        ax.set_ylabel('Value', fontsize=12, fontweight='bold', labelpad=10)
        ax.set_title('Player Comparison with Different Batting Styles', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=10, fontweight='bold')
        
        # Improve legend
        legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                         ncol=5, frameon=True, fontsize=10)
        frame = legend.get_frame()
        frame.set_facecolor('#F3F4F6')
        frame.set_edgecolor('#E5E7EB')
        
        # Add grid only for y-axis and make it subtle
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
        
        # Add subtle background shading to alternate metrics
        for i in range(len(metrics)):
            if i % 2 == 0:
                ax.axvspan(i - 0.5, i + 0.5, color='#F3F4F6', alpha=0.3, zorder=0)
        
    elif player_type == "Bowler":
        # Sample reference values for different player types
        reference_data = {
            "Economist": {"econ": 4.0, "sr": 35, "wickets_per_match": 1.0},
            "Wicket Taker": {"econ": 6.5, "sr": 20, "wickets_per_match": 2.0},
            "Balanced": {"econ": 5.5, "sr": 27, "wickets_per_match": 1.5},
            "Elite": {"econ": 4.5, "sr": 22, "wickets_per_match": 1.8},
            "Current Player": {
                "econ": player_data["econ"], 
                "sr": player_data["sr"], 
                "wickets_per_match": player_data["wickets"] / player_data["mat"] if player_data["mat"] > 0 else 0
            }
        }
        
        # Metrics to compare
        metrics = ["econ", "sr", "wickets_per_match"]
        metric_labels = ["Economy Rate", "Strike Rate", "Wickets per Match"]
        
        # Set x positions and width
        x = np.arange(len(metrics))
        width = 0.15
# Plot bars with improved styling
        for i, (role, values) in enumerate(reference_data.items()):
            offset = width * (i - len(reference_data)/2 + 0.5)
            values_list = [values[m] for m in metrics]
            rects = ax.bar(x + offset, values_list, width, label=role, color=colors[i % len(colors)], 
                            alpha=0.8 if role != "Current Player" else 1.0,
                            edgecolor='white', linewidth=0.7)
            
            # Highlight the current player's bars
            if role == "Current Player":
                for j, v in enumerate(values_list):
                    ax.text(x[j] + offset, v + 0.1, f'{v:.1f}', ha='center', va='bottom', 
                            fontsize=8, fontweight='bold', color=colors[i % len(colors)])
        
        # Customize plot
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold', labelpad=10)
        ax.set_ylabel('Value', fontsize=12, fontweight='bold', labelpad=10)
        ax.set_title('Player Comparison with Different Bowling Styles', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=10, fontweight='bold')
        
        # Improve legend
        legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                            ncol=5, frameon=True, fontsize=10)
        frame = legend.get_frame()
        frame.set_facecolor('#F3F4F6')
        frame.set_edgecolor('#E5E7EB')
        
        # Add grid only for y-axis and make it subtle
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
        
        # Add subtle background shading to alternate metrics
        for i in range(len(metrics)):
            if i % 2 == 0:
                ax.axvspan(i - 0.5, i + 0.5, color='#F3F4F6', alpha=0.3, zorder=0)
    
    plt.tight_layout()
    return fig

# Process analysis if button is clicked
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False

if analyze_button:
    # Show a progress bar for visual feedback
    progress_bar = st.progress(0)
    
    # Simulate processing steps with animation
    for percent_complete in range(101):
        time.sleep(0.01)  # Small delay for animation effect
        progress_bar.progress(percent_complete)
    
    # Prepare player data
    if model_category == "Batsman":
        player_data = {
            "name": player_name if player_name else "Unnamed Batsman",
            "mat": matches,
            "runs": runs,
            "sr": strike_rate,
            "avg": average,
            "boundary_pct": boundary_percent,
            "career_length": career_length
        }
    else:  # Bowler
        player_data = {
            "name": player_name if player_name else "Unnamed Bowler",
            "mat": matches,
            "wickets": wickets,
            "econ": economy,
            "sr": strike_rate,
            "career_length": career_length
        }
    
    # Generate analysis
    st.session_state.analysis_results = generate_analysis(player_data, model_category)
    st.session_state.player_data = player_data
    st.session_state.player_type = model_category
    st.session_state.analyzed = True
    
    # Clear the progress bar
    progress_bar.empty()
    
    # Display a success message with animation
    st.success("Analysis completed successfully!")
    time.sleep(0.5)  # Brief pause
    
    # Switch to Results tab
    st.rerun()

# Display results in the Results tab with improved styling
with tab2:
    if st.session_state.get('analyzed', False):
        player_data = st.session_state.player_data
        analysis_results = st.session_state.analysis_results
        player_type = st.session_state.player_type
        
        # Add a slide-in animation effect for results
        st.markdown("""
        <style>
            @keyframes slideIn {
                0% { opacity: 0; transform: translateY(20px); }
                100% { opacity: 1; transform: translateY(0); }
            }
            .animated-section {
                animation: slideIn 0.6s ease-out forwards;
            }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="animated-section">
            <h2 class='sub-header'>Player Analysis Results</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Player info card with improved styling
        st.markdown("""
        <div class="animated-section" style="animation-delay: 0.2s;">
            <div class='report-header'>Player Information</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div class='report-body'>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"{player_data['name']}")
            st.write(f"Player Type: {player_type}")
            st.write(f"Career Span: {player_data['career_length']} years")
        
        with col2:
            if player_type == "Batsman":
                st.write(f"Matches: {player_data['mat']}")
                st.write(f"Runs: {player_data['runs']}")
                st.write(f"Average: {player_data['avg']}")
                st.write(f"Strike Rate: {player_data['sr']}")
                st.write(f"Boundary %: {player_data['boundary_pct']}%")
            else:  # Bowler
                st.write(f"Matches: {player_data['mat']}")
                st.write(f"Wickets: {player_data['wickets']}")
                st.write(f"Economy: {player_data['econ']}")
                st.write(f"Strike Rate: {player_data['sr']}")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Analysis results with animation delay
        st.markdown("""
        <div class="animated-section" style="animation-delay: 0.4s;">
            <div class='report-header'>AI Analysis</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div class='report-body'>", unsafe_allow_html=True)
        
        # Player classification with highlight
        st.markdown("""
        <h3 style="margin-bottom:0.5rem;">Player Classification</h3>
        <div style="background-color:#EFF6FF; padding:1rem; border-left:4px solid #3B82F6; margin-bottom:1.5rem; border-radius:0.25rem;">
            <span style="font-size:1.2rem; font-weight:600; color:#1E3A8A;">
                {}
            </span>
        </div>
        """.format(analysis_results['prediction']), unsafe_allow_html=True)
        
        # Key insights with improved styling
        st.subheader("Key Insights")
        for i, insight in enumerate(analysis_results['insights']):
            # Add a slight delay effect between items
            st.markdown(f"""
            <div class="recommendation-item" style="animation: fadeIn {0.6 + i*0.1}s ease-in-out;">
                • {insight}
            </div>
            """, unsafe_allow_html=True)
            
        # Visualization with improved layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Performance Radar")
            try:
                fig1 = create_player_visualization(player_data, player_type)
                st.pyplot(fig1)
            except Exception as e:
                st.error(f"Could not create radar visualization: {e}")
        
        with col2:
            st.subheader("Player Comparison")
            try:
                fig2 = create_player_comparison(player_data, player_type)
                st.pyplot(fig2)
            except Exception as e:
                st.error(f"Could not create comparison visualization: {e}")
            
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Recommendations section with animation delay
        st.markdown("""
        <div class="animated-section" style="animation-delay: 0.6s;">
            <div class='report-header'>Recommendations</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div class='report-body'>", unsafe_allow_html=True)
        
        if player_type == "Batsman":
            prediction = analysis_results['prediction'].lower()
            
            recommendations = []
            # Based on player classification
            if "elite" in prediction:
                recommendations = [
                    "Continue with the current approach - excellent performance metrics",
                    "Focus on maintaining consistency across different match conditions",
                    "Consider mentoring younger players to share expertise"
                ]
            elif "anchor" in prediction:
                if "inexperienced" in prediction:
                    recommendations = [
                        "Work on building innings against different bowling types",
                        "Focus on rotating strike more effectively in middle overs",
                        "Develop power-hitting skills for late innings acceleration"
                    ]
                else:
                    recommendations = [
                        "Continue to leverage experience while building partnerships",
                        "Work on tactical aspects against specific bowling types",
                        "Consider improving strike rotation in middle overs"
                    ]
            elif "power" in prediction:
                if "inexperienced" in prediction:
                    recommendations = [
                        "Work on consistency while maintaining aggressive approach",
                        "Develop technique against quality bowling attacks",
                        "Practice situational batting for different match scenarios"
                    ]
                else:
                    recommendations = [
                        "Continue leveraging power-hitting abilities in appropriate situations",
                        "Work on improving shot selection against specific bowlers",
                        "Consider adding more variety to attacking options"
                    ]
            elif "finisher" in prediction:
                if "inexperienced" in prediction:
                    recommendations = [
                        "Develop composure in high-pressure situations",
                        "Practice specific end-game scenarios",
                        "Work on identifying bowlers' variations in death overs"
                    ]
                else:
                    recommendations = [
                        "Continue utilizing experience in pressure situations",
                        "Consider expanding range of shots for specific match situations",
                        "Work on partnership building in run chases"
                    ]
            else:
                recommendations = [
                    "Focus on technical refinement for improved consistency",
                    "Work on specific match scenarios to improve decision making",
                    "Develop a more defined batting identity based on strengths"
                ]
                
            # Display recommendations with animations
            for i, rec in enumerate(recommendations):
                st.markdown(f"""
                <div class="recommendation-item" style="animation: fadeIn {0.8 + i*0.15}s ease-in-out;">
                    <span style="color:#2563EB; font-weight:600;">•</span> {rec}
                </div>
                """, unsafe_allow_html=True)
                
        else:  # Bowler
            prediction = analysis_results['prediction'].lower()
            
            recommendations = []
            if "elite" in prediction:
                recommendations = [
                    "Continue current approach - excellent bowling metrics",
                    "Maintain physical conditioning to ensure longevity",
                    "Consider developing additional variations to stay ahead of batsmen"
                ]
            elif "wicket taker" in prediction:
                if "inexperienced" in prediction:
                    recommendations = [
                        "Work on consistency while maintaining attacking approach",
                        "Develop better control to improve economy rate",
                        "Focus on specific plans for different types of batsmen"
                    ]
                else:
                    recommendations = [
                        "Continue to leverage experience in setting up dismissals",
                        "Work on improving economy rate in certain phases",
                        "Consider adding more variations to your arsenal"
                    ]
            elif "economist" in prediction:
                if "inexperienced" in prediction:
                    recommendations = [
                        "Maintain control while working on wicket-taking deliveries",
                        "Develop specific variations to challenge batsmen",
                        "Practice bowling plans for different match situations"
                    ]
                else:
                    recommendations = [
                        "Continue using experience to contain quality batsmen",
                        "Work on developing more wicket-taking deliveries",
                        "Consider tactical improvements for different phases of the game"
                    ]
            else:
                recommendations = [
                    "Focus on technical refinement for improved consistency",
                    "Work on specific skills like yorkers or slower balls",
                    "Develop a more defined bowling identity based on strengths"
                ]
            
            # Display recommendations with animations
            for i, rec in enumerate(recommendations):
                st.markdown(f"""
                <div class="recommendation-item" style="animation: fadeIn {0.8 + i*0.15}s ease-in-out;">
                    <span style="color:#2563EB; font-weight:600;">•</span> {rec}
                </div>
                """, unsafe_allow_html=True)
                
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Export options with improved styling
        st.markdown("""
        <div class="animated-section" style="animation-delay: 0.8s;">
            <div style="background-color:#F3F4F6; padding:1rem; border-radius:0.5rem; 
                        margin-top:1.5rem; text-align:center; border:1px solid #E5E7EB;">
                <h3 style="margin-bottom:0.8rem; color:#1E3A8A;">Export Analysis</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Centered download button with animation
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.download_button(
                label="Download Analysis Report (CSV)",
                data=pd.DataFrame([player_data]).to_csv(index=False),
                file_name=f"{player_data['name']}_analysis.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        # More attractive placeholder when no analysis has been done
        st.markdown("""
        <div style="background-color:#EFF6FF; padding:2rem; border-radius:0.5rem; 
                    text-align:center; margin-top:2rem; border:1px dashed #60A5FA;">
            <img src="https://www.svgrepo.com/show/474595/analytics.svg" width="80" 
                 style="opacity:0.7; margin-bottom:1rem;">
            <h3 style="color:#1E3A8A; margin-bottom:0.5rem;">No Analysis Results Yet</h3>
            <p style="color:#4B5563;">Please input player statistics and click 'Generate Analysis Report' to view results.</p>
        </div>
        """, unsafe_allow_html=True)

# Improved footer with animation
st.markdown("---")
st.markdown("""
<div class="footer">
    <div style="display:flex; justify-content:center; align-items:center; gap:0.5rem;">
        <span>Cricklytics</span>
        <span style="color:#3B82F6;">•</span>
        <span>Powered by AI</span>
    </div>
    <div style="margin-top:0.5rem; font-size:0.75rem; color:#9CA3AF;">
        © 2025 | Cricket Performance Analytics
    </div>
</div>
""", unsafe_allow_html=True)