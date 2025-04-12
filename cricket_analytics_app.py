import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

# Set page config
st.set_page_config(
    page_title="Cricklytics",
    page_icon="🏏",
    layout="wide"
)

# Custom CSS with improved container styling and animations 
st.markdown("""
<style>
    /* Apply system font stack to all elements */
    body, .stApp, .stApp * {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen-Sans, Ubuntu, Cantarell, 'Helvetica Neue', sans-serif !important;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    /* Main styling and typography */
    * {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        animation: fadeIn 1s ease-in-out;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
    }
    
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2563EB;
        margin-bottom: 0.8rem;
        padding-top: 0.5rem;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
    }
    
    /* Card styling with proper padding and overflow control */
    .card {
        background-color: white;
        border-radius: 0.8rem;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        overflow: hidden;
        word-wrap: break-word;
        border: 1px solid #E2E8F0;
    }
    
    .card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    }
    
    .card-content {
        padding: 0.5rem;
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
        margin-bottom: 0;
        border: none;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
    }
    
    .report-header:hover {
        background-color: #2563EB;
    }
    
    .report-body {
        background-color: white;
        padding: 1.2rem 1.5rem;
        border-radius: 0 0 0.5rem 0.5rem;
        border: 1px solid #E5E7EB;
        border-top: none;
        overflow: hidden;
        word-wrap: break-word;
        margin-bottom: 2rem;
        margin-top: 0;
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
    
    /* Put all content in white boxes */
    .stMarkdown p, .stMarkdown h3, .stMarkdown ul, .stMarkdown ol {
        background-color: white;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin-bottom: 0.5rem;
        border: 1px solid #E5E7EB;
    }
    
    /* Style form inputs */
    div.stNumberInput, div.stTextInput, div.stSlider, div.stSelectbox, div.stMultiSelect {
        background-color: white;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        border: 1px solid #E5E7EB;
    }
    
    /* Fix container backgrounds */
    .stApp {
        background-color: #f1f5f9;
    }
    
    div.block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 95%;
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

# App Header with animation effect
def display_header():
    st.markdown("<h1 class='main-header'>🏏Cricklytics</h1>", unsafe_allow_html=True)
    
    # Add a subheading with animation
    st.markdown(
        """
        <div style="text-align: center; animation: fadeIn 1.5s ease-in-out; margin-bottom: 1.5rem;">
            <p style="font-size: 1.2rem; color: #4B5563;">
                analyse cricket player performance with AI-powered insights from pre-trained models — including a specialized injury prediction engine trained on player workload, match analytics, and recent form.
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )

display_header()

#---------------------------------------------------------------------------
# Player Classification Functions (from cricket_analytics_app.py)
#---------------------------------------------------------------------------

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

# Function to generate insights and analysis for player
def generate_analysis(player_data, player_type):
    try:
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
        # Create a polar subplot
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})

        # Create the radar chart
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # Close the polygon
        angles += angles[:1]

        ax.plot(angles, values, linewidth=2.5, linestyle='solid', color=colors[0])
        ax.fill(angles, values, alpha=0.4, color=colors[2])

        # Axis labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9, fontweight='bold')

        # Title
        ax.set_title(f"Batting Performance - {player_data['name']}", fontsize=13, fontweight='bold', pad=20)

        # Optional: remove radial grid labels if overlapping
        ax.set_yticklabels([])

        # Optional: adjust radial ticks manually
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8], angle=0, fontsize=7)

        # Annotate each point
        for i, v in enumerate(values[:-1]):
            ax.text(angles[i], v + 0.05, f"{int(v * 100)}%", color=colors[0],
                    fontsize=8, ha='center', va='center', fontweight='bold')
            
        
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
        
        # Create a polar subplot
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})

        # Create the radar chart
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # Close the polygon
        angles += angles[:1]

        ax.plot(angles, values, linewidth=2.5, linestyle='solid', color=colors[0])
        ax.fill(angles, values, alpha=0.4, color=colors[2])

        # Axis labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9, fontweight='bold')

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

#---------------------------------------------------------------------------
# Workload Analysis Functions (from workload_analysis.py)
#---------------------------------------------------------------------------

def load_player_data(file, player_name):
    if file is not None:
        df = pd.read_csv(file)
        
        # Convert date column if it exists
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'day' in col.lower()]
        if date_columns:
            for date_col in date_columns:
                try:
                    df[date_col] = pd.to_datetime(df[date_col])
                except:
                    pass
                    
        # Add player identifier
        df['Player'] = player_name
        return df
    return None

def ensure_fatigue_metrics(df):
    # If these metrics already exist, keep them
    # Otherwise, calculate them based on available data
    
    if 'Bowling_Intensity' not in df.columns and all(col in df.columns for col in ['Overs', 'Runs', 'Wick']):
        df['Bowling_Intensity'] = (
            (df['Overs'] * 6) +
            (df['Runs'] * 2) +
            (df['Wick'] * 20)
        )
    
    # Check for match date column
    date_col = None
    for col in df.columns:
        if 'date' in col.lower() or 'day' in col.lower():
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_col = col
                break
    
    # Calculate days since last match if date column exists
    if date_col and 'Days_Since_Last_Match' not in df.columns:
        df = df.sort_values(by=date_col)
        df['Days_Since_Last_Match'] = df.groupby('Player')[date_col].diff().dt.days.fillna(0)
    
    # Calculate cumulative workload if not already present
    if 'Cumulative_Workload' not in df.columns and 'Bowling_Intensity' in df.columns:
        df['Cumulative_Workload'] = df.groupby('Player')['Bowling_Intensity'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
    
    # Calculate workload variance if not already present
    if 'Workload_Variance' not in df.columns and 'Bowling_Intensity' in df.columns:
        df['Workload_Variance'] = df.groupby('Player')['Bowling_Intensity'].rolling(window=3, min_periods=1).std().reset_index(0, drop=True).fillna(0)
    
    return df

#---------------------------------------------------------------------------
# Main Application
#---------------------------------------------------------------------------

# Sidebar for navigation
with st.sidebar:
    st.markdown("<h2 class='sub-header'>Navigation</h2>", unsafe_allow_html=True)
    
    # App features selection
    app_mode = st.radio(
        "Select Analysis Type",
        ["Player Performance Analysis", "Workload & Injury Analysis"]
    )
    
    st.divider()
    
    # Conditional display based on selected mode
    if app_mode == "Player Performance Analysis":
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
    
    else:  # Workload & Injury Analysis
        st.markdown("<h2 class='sub-header'>Upload Player Data</h2>", unsafe_allow_html=True)
        
        player1_file = st.file_uploader("Upload Player 1 Data (CSV)", type=['csv'])
        player2_file = st.file_uploader("Upload Player 2 Data (CSV)", type=['csv'])
        player3_file = st.file_uploader("Upload Player 3 Data (CSV)", type=['csv'])
        
        st.markdown("<div class='model-info'>", unsafe_allow_html=True)
        st.markdown("#### Workload Analysis Info")
        st.write("Upload CSV files with player match data")
        st.write("- Required columns: matches, dates, workload metrics")
        st.write("- Optional: overs, runs, wickets for intensity calculation")
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("### About")
    st.write("This platform combines player performance analysis and workload monitoring to provide comprehensive cricket analytics.")

#---------------------------------------------------------------------------
# Player Performance Analysis Mode
#---------------------------------------------------------------------------

if app_mode == "Player Performance Analysis":
    # Initialize session state variables if not exist
    if 'analysed' not in st.session_state:
        st.session_state.analysed = False
    
    # Create tabs for player analysis
    analysis_tab1, analysis_tab2 = st.tabs(["Player Analysis", "Results"])
    
    with analysis_tab1:
        # Create a unified form with proper styling for player stats input
        st.markdown("<div class='report-card'>", unsafe_allow_html=True)
        st.markdown("<div class='report-header'>Player Statistics Input</div><div class='report-body' style='padding: 1.5rem;'>", unsafe_allow_html=True)
        
        if model_category == "Batsman":
            player_name = st.text_input("Player Name")
            
            col1, col2 = st.columns(2)
            with col1:
                matches = st.number_input("Matches Played", min_value=1, value=50)
            with col2:
                runs = st.number_input("Total Runs", min_value=0, value=1500)
            
            col3, col4 = st.columns(2)
            with col3:
                average = st.number_input("Batting Average", min_value=0.0, value=35.0, format="%.2f")
            with col4:
                strike_rate = st.number_input("Strike Rate", min_value=0.0, value=135.0, format="%.2f")
            
            col5, col6 = st.columns(2)
            with col5:
                fours = st.number_input("Number of Fours", min_value=0, value=120)
            with col6:
                sixes = st.number_input("Number of Sixes", min_value=0, value=45)
            
            col7, col8 = st.columns(2)
            with col7:
                start_year = st.number_input("Career Start Year", min_value=1950, max_value=datetime.now().year, value=2010)
            with col8:
                end_year = st.number_input("Career End Year (Current year if active)", min_value=1950, max_value=datetime.now().year, value=datetime.now().year)
            
            # Calculate boundary percentage automatically
            boundary_pct = ((4 * fours + 6 * sixes) / runs) * 100 if runs > 0 else 0
                
            career_length = end_year - start_year + 1
                
        elif model_category == "Bowler":
            player_name = st.text_input("Player Name")
            col1, col2 = st.columns(2)
            
            with col1:
                matches = st.number_input("Matches Played", min_value=1, value=50)
                wickets = st.number_input("Total Wickets", min_value=0, value=75)
                economy = st.number_input("Economy Rate", min_value=0.0, value=4.5, format="%.2f")
                
            with col2:
                strike_rate = st.number_input("Bowling Strike Rate", min_value=0.0, value=20.0, format="%.2f")
                start_year = st.number_input("Career Start Year", min_value=1950, max_value=datetime.now().year, value=2010)
                end_year = st.number_input("Career End Year (Current year if active)", min_value=1950, max_value=datetime.now().year, value=datetime.now().year)
                
            career_length = end_year - start_year + 1
            
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            analyse_button = st.button("analyse Player", use_container_width=True)
        with col2:
            clear_button = st.button("Clear Data", use_container_width=True)
    
        # Process analysis if button is clicked
        if analyse_button:
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
                    "boundary_pct": boundary_pct,
                    "fours": fours,
                    "sixes": sixes,
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
            st.session_state.analysed = True
            
            # Clear the progress bar
            progress_bar.empty()
            
            # Display a success message with animation
            st.success("Analysis completed successfully!")
            time.sleep(0.5)  # Brief pause
            
            # Switch to Results tab
            st.rerun()
    
    with analysis_tab2:
        if st.session_state.get('analysed', False):
            player_data = st.session_state.player_data
            analysis_results = st.session_state.analysis_results
            player_type = st.session_state.player_type

            # CSS for animations only - we already defined card styles in the main CSS
            st.markdown("""
            <style>
                @keyframes slideIn {
                    0% { opacity: 0; transform: translateY(20px); }
                    100% { opacity: 1; transform: translateY(0); }
                }

                .animated-section {
                    animation: slideIn 0.6s ease-out forwards;
                    margin-bottom: 1.5rem;
                }

                .recommendation-item {
                    font-size: 1rem;
                    padding: 0.25rem 0;
                }

                h3 {
                    margin-top: 0;
                    color: #1E3A8A;
                }
            </style>
            """, unsafe_allow_html=True)

            # Header
            st.markdown("<div class='animated-section'><h2 class='sub-header'>Player Analysis Results</h2></div>", unsafe_allow_html=True)

            # Player Info Card
            st.markdown("<div class='animated-section report-card'>", unsafe_allow_html=True)
            st.markdown("<div class='report-header'>Player Information</div><div class='report-body' style='padding: 1.5rem;'>", unsafe_allow_html=True)

            # Custom CSS for better alignment
            st.markdown("""
            <style>
            .player-stat {
                display: flex;
                margin-bottom: 8px;
                align-items: center;
            }
            .stat-label {
                font-weight: bold;
                min-width: 120px;
                display: inline-block;
            }
            .stat-value {
                flex-grow: 1;
            }
            </style>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""<div class="player-stat">
                    <span class="stat-label">Name:</span> 
                    <span class="stat-value">{player_data['name']}</span>
                </div>""", unsafe_allow_html=True)
                
                st.markdown(f"""<div class="player-stat">
                    <span class="stat-label">Player Type:</span>
                    <span class="stat-value">{player_type}</span>
                </div>""", unsafe_allow_html=True)
                
                st.markdown(f"""<div class="player-stat">
                    <span class="stat-label">Career Span:</span>
                    <span class="stat-value">{player_data['career_length']} years</span>
                </div>""", unsafe_allow_html=True)
            
            with col2:
                if player_type == "Batsman":
                    st.markdown(f"""<div class="player-stat">
                        <span class="stat-label">Matches:</span>
                        <span class="stat-value">{player_data['mat']}</span>
                    </div>""", unsafe_allow_html=True)
                    
                    st.markdown(f"""<div class="player-stat">
                        <span class="stat-label">Runs:</span>
                        <span class="stat-value">{player_data['runs']}</span>
                    </div>""", unsafe_allow_html=True)
                    
                    st.markdown(f"""<div class="player-stat">
                        <span class="stat-label">Average:</span>
                        <span class="stat-value">{player_data['avg']}</span>
                    </div>""", unsafe_allow_html=True)
                    
                    st.markdown(f"""<div class="player-stat">
                        <span class="stat-label">Strike Rate:</span>
                        <span class="stat-value">{player_data['sr']}</span>
                    </div>""", unsafe_allow_html=True)
                    
                    st.markdown(f"""<div class="player-stat">
                        <span class="stat-label">Boundary %:</span>
                        <span class="stat-value">{player_data['boundary_pct']:.2f}%</span>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class="player-stat">
                        <span class="stat-label">Matches:</span>
                        <span class="stat-value">{player_data['mat']}</span>
                    </div>""", unsafe_allow_html=True)
                    
                    st.markdown(f"""<div class="player-stat">
                        <span class="stat-label">Wickets:</span>
                        <span class="stat-value">{player_data['wickets']}</span>
                    </div>""", unsafe_allow_html=True)
                    
                    st.markdown(f"""<div class="player-stat">
                        <span class="stat-label">Economy:</span>
                        <span class="stat-value">{player_data['econ']}</span>
                    </div>""", unsafe_allow_html=True)
                    
                    st.markdown(f"""<div class="player-stat">
                        <span class="stat-label">Strike Rate:</span>
                        <span class="stat-value">{player_data['sr']}</span>
                    </div>""", unsafe_allow_html=True)
            
            # Add space below player information
            st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)

            # AI Analysis Block
            st.markdown("<div class='animated-section report-card'>", unsafe_allow_html=True)
            st.markdown("<div class='report-header'>AI Analysis</div>", unsafe_allow_html=True)

            # Classification Box
            st.markdown(f"""
            <h3>Player Classification</h3>
            <div style="background-color:#EFF6FF; padding:1rem; border-left:4px solid #3B82F6; 
                        margin-bottom:1.5rem; border-radius:0.25rem; color:#1E3A8A; font-weight:600;">
                {analysis_results['prediction']}
            </div>
            """, unsafe_allow_html=True)

            # Key Insights
            st.subheader("Key Insights")
            for i, insight in enumerate(analysis_results['insights']):
                st.markdown(f"""
                <div class="recommendation-item" style="animation: fadeIn {0.6 + i*0.1}s ease-in-out;">
                    • {insight}
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

                
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
            
            # Recommendations section with animation delay
            st.markdown("""
            <div class="animated-section" style="animation-delay: 0.6s;">
                <div class='report-header'>Recommendations</div>
                <div class='report-body' style='padding: 1.5rem;'>
            """, unsafe_allow_html=True)
            
            # Generate recommendations based on player type and stats
            if player_type == "Batsman":
                st.markdown("""
                    ### Training Focus Areas
                    * **Technical Drills**: Focus on improving technique against specific bowling types
                    * **Match Scenarios**: Practice specific game situations based on your player role
                    * **Physical Conditioning**: Targeted fitness plan for batting performance
                    
                    ### Performance Optimization
                    * Continue monitoring strike rate and boundary percentage
                    * analyse match-ups against different bowling styles
                    * Review footage of successful innings to identify patterns
                """)
            else:
                st.markdown("""
                    ### Training Focus Areas
                    * **Skill Refinement**: Work on variations and consistency
                    * **Match Scenarios**: Practice bowling in pressure situations
                    * **Recovery Protocol**: Implement structured recovery to reduce injury risk
                    
                    ### Performance Optimization
                    * Monitor economy rate across different phases
                    * analyse wicket-taking deliveries for patterns
                    * Use data to identify optimal match-ups against batters
                """)
            
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Please fill player statistics and click 'analyse Player' to view results.")

#---------------------------------------------------------------------------
# Workload & Injury Analysis Mode
#---------------------------------------------------------------------------

elif app_mode == "Workload & Injury Analysis":
    st.markdown("<h2 class='sub-header'>Bowler Workload & Injury Prediction Analysis</h2>", unsafe_allow_html=True)
    st.markdown("This dashboard visualizes fatigue metrics across matches and dates for bowlers.")
    
    # Process data
    player1_data = load_player_data(player1_file, "Player 1")
    player2_data = load_player_data(player2_file, "Player 2")
    player3_data = load_player_data(player3_file, "Player 3")
    
    # Combine available data
    player_dfs = [df for df in [player1_data, player2_data, player3_data] if df is not None]
    
    if player_dfs:
        all_data = pd.concat(player_dfs, ignore_index=True)
        all_data = ensure_fatigue_metrics(all_data)
        
        # Data is loaded, show controls in a well-designed card
        st.markdown("<div class='report-card'>", unsafe_allow_html=True)
        st.markdown("<div class='report-header'>Analysis Controls</div>", unsafe_allow_html=True)
        st.markdown("<div class='report-body' style='padding: 1.5rem; background-color: #f8fafc;'>", unsafe_allow_html=True)
        
        available_players = all_data['Player'].unique().tolist()
        selected_players = st.multiselect(
            "Select Players to Display",
            options=available_players,
            default=available_players
        )
        
        # Identify fatigue metrics
        fatigue_metrics = []
        for col in all_data.columns:
            if any(term in col.lower() for term in ['fatigue', 'workload', 'intensity', 'cumulative', 'variance', 'risk']):
                fatigue_metrics.append(col)
        
        # If no fatigue metrics found, suggest alternatives
        if not fatigue_metrics:
            numeric_cols = all_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            for col in numeric_cols:
                if col not in ['Player'] and not any(dim in col.lower() for dim in ['index', 'id']):
                    fatigue_metrics.append(col)
        
        selected_metrics = st.multiselect(
            "Select Fatigue Metrics to Visualize",
            options=fatigue_metrics,
            default=fatigue_metrics[:3] if len(fatigue_metrics) >= 3 else fatigue_metrics
        )
        
        # Identify date/match columns
        date_columns = []
        match_columns = []
        
        for col in all_data.columns:
            if any(term in col.lower() for term in ['date', 'day']):
                if pd.api.types.is_datetime64_any_dtype(all_data[col]):
                    date_columns.append(col)
            if any(term in col.lower() for term in ['match', 'game', 'session']):
                match_columns.append(col)
        
        # If no match column found but we have index, use that
        if not match_columns and 'index' in all_data.columns:
            match_columns.append('index')
        # If still no match columns, create a simple match index
        if not match_columns:
            all_data['Match_Index'] = all_data.groupby('Player').cumcount() + 1
            match_columns.append('Match_Index')
        
        x_axis_options = date_columns + match_columns
        x_axis = st.selectbox(
            "Select X-Axis (Date or Match)",
            options=x_axis_options,
            index=0 if date_columns else 0
        )
        
        st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Filter data based on selection
        filtered_data = all_data[all_data['Player'].isin(selected_players)]
        
        # Create tabs for different visualizations
        workload_tab1, workload_tab2, workload_tab3 = st.tabs(["Metric Trends", "Combined Metrics", "Heatmap Analysis"])
        
        with workload_tab1:
            st.markdown("<h2 class='sub-header'>Fatigue Metrics over Time</h2>", unsafe_allow_html=True)
            
            # Line charts for selected metrics
            for metric in selected_metrics:
                if metric in filtered_data.columns:
                    fig = px.line(filtered_data, x=x_axis, y=metric, color='Player',
                                 title=f"{metric} Over {x_axis} by Player",
                                 markers=True)
                    
                    # Add a horizontal line for average if numeric
                    if pd.api.types.is_numeric_dtype(filtered_data[metric]):
                        avg_value = filtered_data[metric].mean()
                        fig.add_shape(
                            type="line",
                            line=dict(dash="dash", color="gray"),
                            y0=avg_value, y1=avg_value,
                            x0=0, x1=1,
                            xref="paper"
                        )
                        
                    st.plotly_chart(fig, use_container_width=True)
            
        with workload_tab2:
            st.markdown("<h2 class='sub-header'>Combined Metrics Comparison</h2>", unsafe_allow_html=True)
            
            if len(selected_metrics) >= 2:
                # Create subplots - one row per player
                fig = make_subplots(
                    rows=len(selected_players),
                    cols=1,
                    subplot_titles=[f"{player} - Fatigue Metrics" for player in selected_players]
                )
                
                for i, player in enumerate(selected_players):
                    player_data = filtered_data[filtered_data['Player'] == player]
                    
                    for j, metric in enumerate(selected_metrics):
                        if metric in player_data.columns:
                            fig.add_trace(
                                go.Scatter(
                                    x=player_data[x_axis],
                                    y=player_data[metric],
                                    name=f"{player} - {metric}",
                                    mode='lines+markers',
                                    line=dict(dash='solid' if j % 2 == 0 else 'dash')
                                ),
                                row=i+1,
                                col=1
                            )
                
                fig.update_layout(height=300 * len(selected_players), showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Please select at least two metrics to see combined comparison.")
            
        with workload_tab3:
            st.markdown("<h2 class='sub-header'>Fatigue Patterns Heatmap</h2>", unsafe_allow_html=True)
            
            if len(selected_metrics) >= 2:
                # Create a pivot table-like structure for the heatmap
                for player in selected_players:
                    player_data = filtered_data[filtered_data['Player'] == player]
                    
                    # Convert datetime to string for heatmap if needed
                    if pd.api.types.is_datetime64_any_dtype(player_data[x_axis]):
                        x_values = player_data[x_axis].dt.strftime('%Y-%m-%d').values
                    else:
                        x_values = player_data[x_axis].values
                        
                    # Create heatmap data
                    heatmap_data = []
                    for i, metric in enumerate(selected_metrics):
                        if metric in player_data.columns:
                            for j, x_val in enumerate(x_values):
                                heatmap_data.append({
                                    'X': x_val,
                                    'Metric': metric,
                                    'Value': player_data.iloc[j][metric]
                                })
                    
                    if heatmap_data:
                        heatmap_df = pd.DataFrame(heatmap_data)
                        
                        # Create heatmap
                        fig = px.density_heatmap(
                            heatmap_df,
                            x='X',
                            y='Metric',
                            z='Value',
                            color_continuous_scale='Blues',
                            title=f"Fatigue Metrics Heatmap for {player}"
                        )
                        
                        fig.update_layout(
                            xaxis_title=x_axis,
                            yaxis_title="Metric"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Please select at least two metrics to generate heatmap.")
        
        # Show data table with key metrics
        st.markdown("<h2 class='sub-header'>Player Data Table</h2>", unsafe_allow_html=True)
        display_cols = ['Player', x_axis] + selected_metrics
        display_cols = [col for col in display_cols if col in filtered_data.columns]
        st.dataframe(filtered_data[display_cols])
        
        # Injury risk assessment
        st.markdown("<h2 class='sub-header'>Injury Risk Assessment</h2>", unsafe_allow_html=True)
        
        st.markdown("<div class='report-header'>Risk Analysis</div>", unsafe_allow_html=True)
        st.markdown("<div class='report-body' style='padding: 1.5rem; background-color: #f8fafc;'>", unsafe_allow_html=True)
        
        for player in selected_players:
            player_data = filtered_data[filtered_data['Player'] == player]
            
            if 'Cumulative_Workload' in player_data.columns or 'Bowling_Intensity' in player_data.columns:
                # Calculate simple risk score
                workload_metric = 'Cumulative_Workload' if 'Cumulative_Workload' in player_data.columns else 'Bowling_Intensity'
                
                if len(player_data) > 0:
                    recent_workload = player_data[workload_metric].iloc[-1] if not player_data.empty else 0
                    avg_workload = player_data[workload_metric].mean() if not player_data.empty else 0
                    
                    # Calculate risk score (simplified)
                    risk_score = (recent_workload / avg_workload * 100) if avg_workload > 0 else 50
                    
                    # Risk categories
                    if risk_score > 130:
                        risk_category = "High"
                        risk_color = "red"
                    elif risk_score > 110:
                        risk_category = "Moderate"
                        risk_color = "orange"
                    else:
                        risk_category = "Low"
                        risk_color = "green"
                    
                    st.markdown(f"### {player}")
                    st.markdown(f"Risk Category: <span style='color:{risk_color};font-weight:bold;'>{risk_category}</span>", unsafe_allow_html=True)
                    
                    # Create progress bar for visualization
                    st.progress(min(risk_score/200, 1.0))
                    
                    # Add recommendations
                    if risk_category == "High":
                        st.markdown("**Recommendations:**")
                        st.markdown("- Consider resting player for upcoming matches")
                        st.markdown("- Implement reduced bowling workload")
                        st.markdown("- Schedule additional recovery sessions")
                    elif risk_category == "Moderate":
                        st.markdown("**Recommendations:**")
                        st.markdown("- Monitor player closely during matches")
                        st.markdown("- Consider reduced bowling spell lengths")
                        st.markdown("- Increase recovery protocols")
                    else:
                        st.markdown("**Recommendations:**")
                        st.markdown("- Maintain current workload management")
                        st.markdown("- Continue regular monitoring")
                    
                    st.markdown("---")
            else:
                st.markdown(f"### {player}")
                st.markdown("Insufficient data to calculate injury risk.")
                st.markdown("---")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    else:
        st.info("Please upload data files for at least one player to begin visualization.")
        
        # Show sample data format
        st.markdown("<h2 class='sub-header'>Sample Data Format</h2>", unsafe_allow_html=True)
        st.markdown("""
        Upload CSV files with the following columns:
        - **Date**: Match date (YYYY-MM-DD format)
        - **Overs**: Number of overs bowled
        - **Runs**: Runs conceded
        - **Wick**: Wickets taken
        - **Bowling_Intensity**: (Optional) Calculated from overs, runs, and wickets
        - **Cumulative_Workload**: (Optional) Rolling average of bowling intensity
        - **Workload_Variance**: (Optional) Rolling standard deviation of bowling intensity
        
        The system will calculate missing metrics if not provided.
        """)
        
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