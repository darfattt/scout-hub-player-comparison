import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import glob
import os
import re
import io
import logging
import matplotlib.image as mpimg
from utils import load_css, get_player_image
from data_utils import extract_player_name, extract_player_info, ensure_numeric_columns, calculate_percentile_ranks
from visualization import generate_unified_player_chart, generate_radar_chart, generate_forward_type_scatter

# Try to import scipy, provide fallback if not available
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # Define a simple fallback function for percentile calculation
    class StatsFallback:
        @staticmethod
        def percentileofscore(data, value):
            """Simple fallback for scipy.stats.percentileofscore."""
            if not data:
                return 50
            
            count = sum(1 for x in data if x <= value)
            return (count / len(data)) * 100
    
    stats = StatsFallback()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("player_comparison_tool")

# Configure matplotlib for better visualization
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.titlesize': 11,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 12,
    'figure.dpi': 120,
    'savefig.dpi': 300,
    'figure.figsize': (8, 6),
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3
})

# Set page config
st.set_page_config(
    page_title="Player Comparison Tool",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load custom CSS
st.markdown(f'<style>{load_css("styles.css")}</style>', unsafe_allow_html=True)

# Add custom CSS to ensure full width layout
st.markdown("""
<style>
    .block-container {
        max-width: 100% !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-top: 0.5rem !important;
    }
    .main .block-container {
        padding: 1rem !important;
    }
    /* Remove gaps between elements */
    .row-widget.stHorizontal {
        gap: 5px !important;
    }
    /* Ensure tables take full width */
    .stats-table-container {
        width: 100% !important;
        margin: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Main app
def main():
    st.markdown('<p class="title">Scouting Tools</p>', unsafe_allow_html=True)
    
    # Load data from the CSV files
    csv_files = glob.glob('data/wyscout/*.csv')
    
    logger.info(f"Starting application - found {len(csv_files)} CSV files")
    
    if not csv_files:
        st.warning("No player data files found. This application requires Wyscout data in CSV format.")
        
        # Display instructions for adding data
        st.markdown("""
        ## How to Add Player Data
        
        This application requires player statistics files in CSV format from Wyscout or similar providers.
        
        To add player data:
        
        1. Place your CSV files in the `data/wyscout/` directory
        2. Files should be named: `Player stats <Player Name>.csv`
        3. Restart the application
        
        ### Sample Data Structure
        
        Your CSV files should contain columns such as:
        - Date, Match, Competition (for match identification)
        - Position (player's position)
        - Various statistics (Goals, Assists, Passes, etc.)
        
        If you're using this in Streamlit Cloud, you'll need to:
        1. Fork the repository
        2. Add your data files to the repository
        3. Deploy your own version of the app
        """)
        
        # Add option to use sample data
        if st.button("Use Demo Data"):
            # Create sample data directory if it doesn't exist
            os.makedirs('data/wyscout', exist_ok=True)
            
            # Create sample player data
            sample_players = ["Sample Player 1", "Sample Player 2", "Sample Player 3"]
            
            for player in sample_players:
                # Create a simple dataframe with basic stats
                sample_df = pd.DataFrame({
                    'Date': ['2023-01-01', '2023-01-08', '2023-01-15'],
                    'Match': [f'Match {i+1}' for i in range(3)],
                    'Competition': ['League A', 'League A', 'Cup'],
                    'Position': ['Forward', 'Forward', 'Forward'],
                    'Minutes played': [90, 85, 90],
                    'Goals': [1, 0, 2],
                    'Assists': [0, 1, 0],
                    'Shots': [3, 2, 4],
                    'Passes accurate': [25, 30, 22],
                    'Dribbles successful': [4, 5, 3],
                    'Duels won': [8, 7, 9],
                    'Recoveries': [5, 6, 4]
                })
                
                # Save to CSV
                sample_file = f'data/wyscout/Player stats {player}.csv'
                sample_df.to_csv(sample_file, index=False)
                
                logger.info(f"Created sample data file: {sample_file}")
                
            st.success("Sample data has been created. Please refresh the page to load it.")
            
        return
        
    player_dfs = []
    player_names = []
    player_files = {}  # Dictionary to map player names to file paths
    player_latest_dates = {}  # Dictionary to track latest match date for each player
    
    with st.spinner("Loading player data..."):
        for file in csv_files:
            try:
                player_name = extract_player_name(file)
                player_names.append(player_name)
                player_files[player_name] = file
                
                # Try to determine the latest match date for this player
                try:
                    df = pd.read_csv(file, encoding='latin1')
                    if 'Date' in df.columns and not df['Date'].empty:
                        # Try different date formats
                        for date_format in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%Y/%m/%d']:
                            try:
                                df['parsed_date'] = pd.to_datetime(df['Date'], format=date_format)
                                latest_date = df['parsed_date'].max()
                                player_latest_dates[player_name] = latest_date
                                break
                            except:
                                continue
                except Exception as e:
                    logger.warning(f"Could not determine latest date for {player_name}: {str(e)}")
                
                logger.info(f"Found player data file for {player_name}: {file}")
                #st.success(f"Found player data for {player_name}")
            except Exception as e:
                logger.error(f"Error processing {file}: {str(e)}", exc_info=True)
                st.error(f"Error processing {file}: {str(e)}")
    
    # Create a configuration section
    with st.expander("⚙️ Configuration", expanded=True):
        st.markdown("""
        <div style=" border-radius: 8px; padding: 10px; margin-bottom: 10px;">
            <p style="margin: 0; font-size: 14px;">Select players to compare and customize visualization settings</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sort players by latest match date if available
        if player_latest_dates:
            # Sort players by latest date (most recent first)
            sorted_players = sorted(
                [(name, player_latest_dates.get(name, pd.Timestamp('1970-01-01'))) for name in player_names],
                key=lambda x: x[1],
                reverse=True
            )
            # Extract just the names in sorted order
            sorted_player_names = [name for name, _ in sorted_players]
            # Get the 3 most recent players
            latest_players = sorted_player_names[:3]
            default_selections = latest_players if len(latest_players) <= 3 else []
        else:
            # Fallback to original behavior if dates couldn't be determined
            default_selections = player_names[:3] if len(player_names) <= 3 else []
        
        # Allow user to select players to compare (multi-select with default to 3 latest players)
        selected_players = st.multiselect(
            "Select players to compare (max 3):",
            options=player_names,
            default=default_selections,
            max_selections=3
        )
        
        if not selected_players:
            st.warning("Please select at least one player to continue.")
            logger.warning("No players selected, waiting for user input")
            return
        
        logger.info(f"User selected players: {', '.join(selected_players)}")
        
        # Add Per90 mode toggle
        use_per90 = st.checkbox("Use Per 90 Minutes Statistics", value=True, 
                               help="Calculate all statistics per 90 minutes of play instead of per match")
        
        if use_per90:
            st.info("""
            **Per 90 Minutes Mode**: Statistics will be normalized to a 'per 90 minutes' basis for fair comparison.
            
            This mode:
            - Adjusts all numeric stats based on actual minutes played (90/minutes * stat_value)
            - Accounts for substitutions and partial appearances
            - Makes comparisons fairer between players with different playing times
            - Helps identify productive players who may have limited minutes
            """)
            logger.info("Per90 mode enabled")
        else:
            logger.info("Using raw statistics (per match)")
        
        # Load the selected players' raw data
        raw_player_dfs = {}
        for player in selected_players:
            try:
                logger.info(f"Loading data for player: {player}")
                df = pd.read_csv(player_files[player], encoding='latin1')
                
                # Special preprocessing for card data
                if 'Yellow card' in df.columns:
                    # Clean up card values
                    df['Yellow card'] = df['Yellow card'].apply(lambda x: 
                        pd.to_numeric(x, errors='coerce') if not isinstance(x, str) else
                        (float(x.strip()) if x.strip().replace('.', '', 1).isdigit() else 
                         (1 if x.strip() and x.strip() not in ['0', '0.0'] else 0)))
                
                if 'Red card' in df.columns:
                    # Clean up card values  
                    df['Red card'] = df['Red card'].apply(lambda x: 
                        pd.to_numeric(x, errors='coerce') if not isinstance(x, str) else
                        (float(x.strip()) if x.strip().replace('.', '', 1).isdigit() else 
                         (1 if x.strip() and x.strip() not in ['0', '0.0'] else 0)))
                
                raw_player_dfs[player] = df
                logger.info(f"Successfully loaded raw data for {player}: {df.shape[0]} rows, {df.shape[1]} columns")
            except UnicodeDecodeError:
                # Try different encodings if latin1 fails
                logger.warning(f"UnicodeDecodeError with latin1 encoding for {player}, trying utf-8")
                try:
                    df = pd.read_csv(player_files[player], encoding='utf-8')
                    raw_player_dfs[player] = df
                    logger.info(f"Successfully loaded raw data with utf-8 encoding for {player}")
                except Exception as e:
                    logger.error(f"Error loading raw data for {player}: {str(e)}", exc_info=True)
                    st.error(f"Error loading raw data for {player}: {str(e)}")
            except Exception as e:
                logger.error(f"Error loading raw data for {player}: {str(e)}", exc_info=True)
                st.error(f"Error loading raw data for {player}: {str(e)}")
        
        # Extract all available competitions for each player
        all_competitions = {}
        for player, df in raw_player_dfs.items():
            if 'Competition' in df.columns:
                player_competitions = df['Competition'].dropna().unique().tolist()
                all_competitions[player] = player_competitions
        
        # Visualization options
        viz_type = st.radio(
            "Visualization type:", #, "Radar Chart"
            options=["Bar Chart"],
            horizontal=True
        )
    
        logger.info(f"Selected visualization type: {viz_type}")
        
        # Category selection
        available_categories = ["Defensive", "Progressive", "Offensive", "Goalkeeping", "Distribution"]
        selected_categories = st.multiselect(
            "Filter metrics by category:",
            options=available_categories,
            default=available_categories
        )
        logger.info(f"Selected categories: {', '.join(selected_categories)}")
        
        # Create filtered dataframes based on competition selection
        selected_dfs = []
        competition_filters = {}
        
        # Create columns for competition filters
        filter_cols = st.columns(len(selected_players))
        
        for i, (player, col) in enumerate(zip(selected_players, filter_cols)):
            with col:
                st.markdown(f"<p style='font-weight: bold; font-size: 14px;'>{player} Competitions</p>", unsafe_allow_html=True)
                
                if player in all_competitions and all_competitions[player]:
                    # Add "All" option at the beginning
                    competition_options = ["All"] + all_competitions[player]
                    
                    # Try to determine the latest competition
                    latest_competition = "All"
                    try:
                        if player in raw_player_dfs and 'Date' in raw_player_dfs[player].columns and 'Competition' in raw_player_dfs[player].columns:
                            # Convert date column to datetime, handle various formats
                            df = raw_player_dfs[player].copy()
                            try:
                                # Try to find the most recent competition based on date
                                if not df['Date'].empty:
                                    # Try different date formats
                                    for date_format in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%Y/%m/%d']:
                                        try:
                                            df['parsed_date'] = pd.to_datetime(df['Date'], format=date_format)
                                            break
                                        except:
                                            continue
                                    
                                    # If successful in parsing dates
                                    if 'parsed_date' in df.columns:
                                        # Get the latest date and its corresponding competition
                                        latest_data = df.sort_values('parsed_date', ascending=False).iloc[0]
                                        if 'Competition' in latest_data and pd.notna(latest_data['Competition']):
                                            latest_competition = latest_data['Competition']
                                            if latest_competition not in competition_options:
                                                latest_competition = "All"
                            except Exception as e:
                                logger.warning(f"Could not determine latest competition for {player}: {str(e)}")
                                latest_competition = "All"
                    except Exception as e:
                        logger.warning(f"Error getting latest competition for {player}: {str(e)}")
                        latest_competition = "All"
                    
                    # Set the default selection to the latest competition or "All" if not found
                    default_competition = [latest_competition]
                    
                    selected_competitions = st.multiselect(
                        f"Filter by competition:",
                        options=competition_options,
                        default=default_competition,
                        key=f"competition_{player}",
                        label_visibility="collapsed"
                    )
                    
                    # If no selections made, default to the latest competition
                    if not selected_competitions:
                        selected_competitions = default_competition
                    
                    competition_filters[player] = selected_competitions
                    logger.info(f"Selected competitions for {player}: {', '.join(selected_competitions)}")
                    
                    # Filter the dataframe based on competition selection
                    if player in raw_player_dfs:
                        df = raw_player_dfs[player].copy()
                        
                        if "All" not in selected_competitions and 'Competition' in df.columns:
                            # Filter to include only the selected competitions
                            df = df[df['Competition'].isin(selected_competitions)]
                            logger.info(f"Filtered {player} data to competitions '{', '.join(selected_competitions)}': {df.shape[0]} rows")
                        
                        # Replace NaN values with 0 for numeric columns
                        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                        df[numeric_cols] = df[numeric_cols].fillna(0)
                        
                        # Apply per90 normalization if enabled
                        if use_per90 and 'Minutes played' in df.columns:
                            logger.info(f"Applying per90 normalization for {player}")
                            # Get minutes played for each match
                            df['Minutes played'] = pd.to_numeric(df['Minutes played'], errors='coerce')
                            
                            # Calculate normalization factor for each match (90 / minutes played)
                            # Cap at reasonable values to avoid extreme normalization for very short appearances
                            df['per90_factor'] = 90 / df['Minutes played'].clip(lower=15, upper=None)
                            
                            # Apply normalization to all numeric columns except Minutes played
                            for col in numeric_cols:
                                if col != 'Minutes played':
                                    df[col] = df[col] * df['per90_factor']
                        
                        df['player_name'] = player  # Add player name to dataframe
                        selected_dfs.append(df)
                else:
                    st.text("No competition data available")
                    
                    # Use the full dataset if no competitions found
                    if player in raw_player_dfs:
                        df = raw_player_dfs[player].copy()
                        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                        df[numeric_cols] = df[numeric_cols].fillna(0)
                        
                        # Apply per90 normalization if enabled
                        if use_per90 and 'Minutes played' in df.columns:
                            logger.info(f"Applying per90 normalization for {player}")
                            # Get minutes played for each match
                            df['Minutes played'] = pd.to_numeric(df['Minutes played'], errors='coerce')
                            
                            # Calculate normalization factor for each match (90 / minutes played)
                            # Cap at reasonable values to avoid extreme normalization for very short appearances
                            df['per90_factor'] = 90 / df['Minutes played'].clip(lower=15, upper=None)
                            
                            # Apply normalization to all numeric columns except Minutes played
                            for col in numeric_cols:
                                if col != 'Minutes played':
                                    df[col] = df[col] * df['per90_factor']
                        
                        df['player_name'] = player
                        selected_dfs.append(df)
    
    # Extract player info
    logger.info("Extracting player info")
    player_info = []
    for i, df in enumerate(selected_dfs):
        info = extract_player_info(df)
        player_info.append(info)
        logger.info(f"Player {selected_players[i]} info: {info}")
    
    # Define colors for players (similar to the reference image)
    player_colors = ['#CD5C5C', '#4169E1', '#228B22']  # Red, Blue, Green
    
    # Define stat categories
    stat_categories = {
        "General": [
            "Minutes played", 
            "Total actions",
            "Total actions successful",
            "Match",
            "Competition",
            "Date",
            "Position"
        ],
        "Defensive": [
            "Duels",
            "Duels won", 
            "Aerial duels", 
            "Aerial duels won", 
            "Interceptions", 
            "Losses", 
            "Losses own half", 
            "Recoveries", 
            "Recoveries opp. half"
        ],
        "Progressive": [
            "Passes",
            "Passes accurate", 
            "Long passes", 
            "Long passes accurate", 
            "Crosses", 
            "Crosses accurate", 
            "Dribbles",
            "Dribbles successful"
        ],
        "Offensive": [
            "Goals", 
            "Assists", 
            "Shots", 
            "Shots On Target", 
            "xG"
        ],
        "Goalkeeping": [
            "Conceded goals",
            "xCG",
            "Shots against",
            "Saves",
            "Saves with reflexes",
            "Exits"
        ],
        "Distribution": [
            "Long passes",
            "Long passes accurate",
            "Short passes",
            "Short passes accurate",
            "Goal kicks",
            "Short goal kicks",
            "Long goal kicks"
        ]
    }
    
    # Filter categories based on user selection
    filtered_stat_categories = {k: v for k, v in stat_categories.items() 
                              if k == "General" or k in selected_categories}
    
    # Filter stats based on selected category
    filtered_categories = {}
    for category, stats in filtered_stat_categories.items():
        available_stats = []
        for stat in stats:
            # Skip Yellow card and Red card statistics
            if stat in ["Yellow card", "Red card"]:
                continue
            
            for df in selected_dfs:
                if stat in df.columns:
                    available_stats.append(stat)
                    break
        filtered_categories[category] = available_stats
    
    # Get all unique stats across categories
    all_stats = []
    for stats in filtered_categories.values():
        all_stats.extend(stats)
    all_stats = list(set(all_stats))
    
    if not all_stats:
        st.warning("No common statistics found across selected players. Please check your CSV files.")
        return
    
    # Check which stats are actually available and numeric in the data
    available_numeric_stats = []
    non_numeric_columns = ["Match", "Competition", "Date", "Position"]
    
    # First, check which stats are common across all datasets
    common_stats = set(all_stats)
    for df in selected_dfs:
        common_stats &= set(df.columns)
    
    # If no common stats exist, show warning and return
    if not common_stats:
        st.warning("No common statistics found across selected players. Please check your CSV files.")
        return
    
    # Convert each dataframe to only include numeric columns
    numeric_dfs = []
    for df in selected_dfs:
        numeric_df = ensure_numeric_columns(df, non_numeric_columns)
        numeric_dfs.append(numeric_df)
    
    # Get the common numeric columns across all dataframes
    numeric_cols = set(numeric_dfs[0].columns)
    for df in numeric_dfs[1:]:
        numeric_cols &= set(df.columns)
    
    # Filter for stats that were originally requested and are numeric
    available_numeric_stats = [stat for stat in all_stats 
                             if stat in numeric_cols and stat not in ["Yellow card", "Red card"]]
    
    # Log the available numeric stats
    logger.info(f"Found {len(available_numeric_stats)} numeric stats for analysis: {', '.join(available_numeric_stats)}")
    
    if not available_numeric_stats:
        st.warning("No common numeric statistics found. Please check your CSV files for numeric metrics.")
        logger.warning("No common numeric statistics found in the data")
        return
    
    # Calculate percentile ranks
    logger.info("Calculating percentile ranks")
    player_percentiles, player_actual_values = calculate_percentile_ranks(selected_dfs, available_numeric_stats)
    
    if not player_percentiles:
        st.warning("Could not calculate percentile ranks. Please ensure all players have at least some common statistics.")
        return
    
    # Get custom player colors
    
    # Create visualization based on selected type
    if viz_type == "Bar Chart":
        # Create columns to display player charts based on number of selected players
        num_players = len(selected_players)
        cols = st.columns(num_players)
        
        # Create and display unified charts for each player
        for i, (name, percentile_df, actual_values, col) in enumerate(zip(selected_players, player_percentiles, player_actual_values, cols)):
            with col:
                # Get player image path
                player_image_path = get_player_image(name)
                
                fig = generate_unified_player_chart(
                    name, 
                    percentile_df, 
                    player_colors[i % len(player_colors)], 
                    player_info[i], 
                    player_image_path,
                    actual_values
                )
                
                if fig:
                    st.pyplot(fig)
                    
                    # Add download button for this specific chart
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                    buf.seek(0)
                    
                    st.download_button(
                        label="📥 Download Chart",
                        data=buf,
                        file_name=f"{name}_stats_chart.png",
                        mime="image/png"
                    )
                else:
                    st.warning(f"Could not generate chart for {name}. Insufficient data.")
    else:
        # Generate radar chart for all selected players
        st.markdown("#### Radar Chart Comparison")
        radar_fig = generate_radar_chart(selected_players, player_percentiles, player_colors, player_actual_values)
        st.pyplot(radar_fig)
        
        # Add download button for radar chart
        buf = io.BytesIO()
        radar_fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        
        st.download_button(
            label="📥 Download Radar Chart",
            data=buf,
            file_name="player_comparison_radar.png",
            mime="image/png"
        )
    
    # --- Forward Role Profile Scores ---
    st.markdown('<div class="stats-table-container" style="margin-bottom: 30px;">', unsafe_allow_html=True)
    
    # Add per90 indication if enabled
    if use_per90:
        st.markdown('<p class="stats-table-header">Forward Role Profile Scores (Per 90 Minutes)</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="stats-table-header">Forward Role Profile Scores</p>', unsafe_allow_html=True)
    

    # Define metrics and weights for each forward type
    forward_role_weights = {
        "Advance Forward": {
            "Goals": 0.3,
            "Shots": 0.2,
            "xG": 0.15,
            "Dribbles successful": 0.15,
            "Passes Received": 0.1,
            "Touches Att 3rd": 0.1
        },
        "Pressing Forward": {
            "Duels won": 0.25,
            "Recoveries": 0.2,
            "Pressures": 0.2,
            "Goals": 0.15,
            "Shots": 0.1,
            "Interceptions": 0.1
        },
        "Deep-lying Forward": {
            "SCA": 0.25,
            "xAG": 0.25,
            "Key Passes": 0.15,
            "Progressive Passes Received": 0.15,
            "Touches Att 3rd": 0.1,
            "npxG": 0.1
        },
        "Poacher": {
            "npxG": 0.5,
            "Shots": 0.5,
            "Shots on Target": 0.1,
            "Offsides": 0.1,
            "Average Shot Distance": -0.1,
            "Touches": -0.05,
            "Passes Received": -0.05
        }
    }

    # Define metrics and weights for goalkeeper roles
    goalkeeper_role_weights = {
        "Shot Stopper": {
            "Saves": 0.3,
            "Saves with reflexes": 0.25,
            "Conceded goals": -0.2,
            "xCG": -0.15,
            "Shots against": 0.1
        },
        "Sweeper Keeper": {
            "Exits": 0.25,
            "Recoveries": 0.2,
            "Long passes accurate": 0.2,
            "Short passes accurate": 0.15,
            "Goal kicks": 0.1,
            "Short goal kicks": 0.05,
            "Long goal kicks": 0.05
        }
    }

    # Combine role weights based on position
    role_weights = {
        "GK": goalkeeper_role_weights,
        "Forward": forward_role_weights
    }

    # Normalize and score each player for each role
    def compute_role_scores(player_actual_values, available_stats, weights):
        # Gather all unique stats from all roles
        all_stats = set()
        for position_weights in weights.values():
            for role_weights in position_weights.values():
                all_stats.update(role_weights.keys())
        
        # Normalize stats (min-max across all players for each stat)
        stat_min = {stat: min([float(df[stat].iloc[0]) if stat in df.columns else 0 for df in player_actual_values]) for stat in all_stats}
        stat_max = {stat: max([float(df[stat].iloc[0]) if stat in df.columns else 0 for df in player_actual_values]) for stat in all_stats}
        
        scores = []
        for i, df in enumerate(player_actual_values):
            player_score = {}
            
            # Get player position
            position = player_info[i].get('position', 'Forward')
            
            # Get appropriate role weights based on position
            position_weights = weights.get(position, weights.get('Forward'))  # Default to Forward if position not found
            
            for role, role_weights in position_weights.items():
                score = 0
                for stat, w in role_weights.items():
                    if stat in df.columns:
                        val = float(df[stat].iloc[0])
                        # Min-max normalization
                        if stat_max[stat] != stat_min[stat]:
                            norm = (val - stat_min[stat]) / (stat_max[stat] - stat_min[stat])
                        else:
                            norm = 0.5  # If all values are the same
                        score += w * norm
                player_score[role] = score
            scores.append(player_score)
        return scores

    # Compute scores for all players
    role_scores = compute_role_scores(player_actual_values, available_numeric_stats, role_weights)
    
    # Get appropriate role names based on player position
    role_names = []
    for i, player in enumerate(selected_players):
        position = player_info[i].get('position', 'Forward')
        if position == 'GK':
            role_names.append(list(goalkeeper_role_weights.keys()))
        else:
            role_names.append(list(forward_role_weights.keys()))
    
    # Create a row for player profiles
    player_cols = st.columns(len(selected_players))
    
    # Create a profile chart for each player
    for i, (player, player_score, col) in enumerate(zip(selected_players, role_scores, player_cols)):
        with col:
            # Create player title with info
            player_position = player_info[i].get('position', 'Forward')
            player_matches = player_info[i].get('total_matches', 0)
            st.markdown(f"<h3 style='text-align: center; margin-bottom: 10px; font-size: 16px; color: white;'>{player} ({player_position})</h3>", unsafe_allow_html=True)
            
            # Create separate figure for this player
            fig = go.Figure()
            
            # Sort role scores for this player from highest to lowest
            sorted_roles = sorted([(role, player_score[role]) for role in role_names[i]], key=lambda x: x[1], reverse=True)
            role_labels = [role for role, _ in sorted_roles]
            role_values = [score for _, score in sorted_roles]
            
            # Add trace for horizontal bar
            fig.add_trace(go.Bar(
                y=role_labels,
                x=role_values,
                orientation='h',
                marker=dict(
                    color=player_colors[i % len(player_colors)],
                    line=dict(width=1, color='#222')
                ),
                text=[f"{value:.2f}" for value in role_values],
                textposition='auto',
                textfont=dict(color='white'),
                showlegend=False
            ))
            
            # Update layout
            fig.update_layout(
                title=dict(
                    text=f"Profile Score Distribution",
                    font=dict(size=14, color='white'),
                    x=0.5
                ),
                plot_bgcolor='#222',
                paper_bgcolor='#222',
                height=300,
                margin=dict(l=15, r=15, t=40, b=20),
                xaxis=dict(
                    title='Score',
                    showgrid=True,
                    gridcolor='#444',
                    tickfont=dict(size=10, color='#CCC')
                ),
                yaxis=dict(
                    title='',
                    tickfont=dict(size=12, color='#FFF'),
                    automargin=True
                ),
                font=dict(color='#EEE')
            )
            
            # Display chart
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Add info message only for non-goalkeeper positions
    has_goalkeeper = any(info.get('position') == 'GK' for info in player_info)
    
    if not has_goalkeeper:
        st.info(
            """
Each player is scored for four classic forward roles based on their stats and the latest competition filter:

- **Advance Forward**: Direct goal threat, excels at finishing and movement.
- **Pressing Forward**: High work rate, presses defenders, wins duels.
- **Deep-lying Forward**: Drops deep, creates chances, links play.
- **Poacher**: Focuses on scoring, operates in the box, exploits chances.
            """
        )
    else:
        st.info(
            """
Each goalkeeper is scored for three classic goalkeeper roles based on their stats and the latest competition filter:

- **Shot Stopper**: Excels at making saves and preventing goals.
- **Sweeper Keeper**: Reads the game well and acts as an extra defender.
            """
        )

    # Add detailed weight information in a collapsible section
    with st.expander("📊 View Role Weight Details", expanded=False):
        st.markdown("""
        ### Role Weight Details
        
        Each player role is defined by a weighted combination of key statistics that determine the player's suitability for that role. 
        The weights below show which statistics are most important for each role type.
        
        #### Forward Roles
        """)
        
        # Create a table for Advance Forward weights
        af_data = [[stat, weight] for stat, weight in forward_role_weights["Advance Forward"].items()]
        af_df = pd.DataFrame(af_data, columns=["Statistic", "Weight"])
        st.dataframe(af_df.style.format({"Weight": "{:.2f}"}), use_container_width=True)
        
        st.markdown("#### Pressing Forward")
        pf_data = [[stat, weight] for stat, weight in forward_role_weights["Pressing Forward"].items()]
        pf_df = pd.DataFrame(pf_data, columns=["Statistic", "Weight"])
        st.dataframe(pf_df.style.format({"Weight": "{:.2f}"}), use_container_width=True)
        
        st.markdown("#### Deep-lying Forward")
        dlf_data = [[stat, weight] for stat, weight in forward_role_weights["Deep-lying Forward"].items()]
        dlf_df = pd.DataFrame(dlf_data, columns=["Statistic", "Weight"])
        st.dataframe(dlf_df.style.format({"Weight": "{:.2f}"}), use_container_width=True)
        
        st.markdown("#### Poacher")
        poacher_data = [[stat, weight] for stat, weight in forward_role_weights["Poacher"].items()]
        poacher_df = pd.DataFrame(poacher_data, columns=["Statistic", "Weight"])
        # Format the weights to show negative values with a minus sign
        poacher_formatted = poacher_df.style.format({"Weight": "{:.2f}"})
        st.dataframe(poacher_formatted, use_container_width=True)

        st.markdown("""
        #### Goalkeeper Roles
        """)

        st.markdown("#### Shot Stopper")
        ss_data = [[stat, weight] for stat, weight in goalkeeper_role_weights["Shot Stopper"].items()]
        ss_df = pd.DataFrame(ss_data, columns=["Statistic", "Weight"])
        st.dataframe(ss_df.style.format({"Weight": "{:.2f}"}), use_container_width=True)

        st.markdown("#### Sweeper Keeper")
        sk_data = [[stat, weight] for stat, weight in goalkeeper_role_weights["Sweeper Keeper"].items()]
        sk_df = pd.DataFrame(sk_data, columns=["Statistic", "Weight"])
        st.dataframe(sk_df.style.format({"Weight": "{:.2f}"}), use_container_width=True)

        
        st.markdown("""
        **Note on weights:**
        - Positive weights (most stats) indicate that higher values contribute more to the role score
        - Negative weights (in some roles) indicate that lower values are better for that role
        - The higher the weight value, the more important that statistic is for the role
        
        *These weights can be adjusted based on tactical preferences or analysis requirements.*
        """)

    # Add new table view for player stats comparison
    st.markdown('<div class="stats-table-container">', unsafe_allow_html=True)
    
    # Add per90 indication to table header if enabled
    if use_per90:
        st.markdown('<p class="stats-table-header">Player Statistics Comparison Table (Per 90 Minutes)</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="stats-table-header">Player Statistics Comparison Table</p>', unsafe_allow_html=True)
    
    # Create category filters for the table view
    table_category_options = ["All"] + list(filtered_stat_categories.keys())
    selected_table_category = st.radio(
        "Filter table metrics by category:",
        options=table_category_options,
        horizontal=True
    )
    
    # Filter stats based on selected category
    filtered_table_stats = []
    if selected_table_category == "All":
        filtered_table_stats = [stat for stat in available_numeric_stats if stat not in ["Yellow card", "Red card"]]
    else:
        # Get stats from the selected category
        for stat in available_numeric_stats:
            if stat in filtered_stat_categories.get(selected_table_category, []) and stat not in ["Yellow card", "Red card"]:
                filtered_table_stats.append(stat)
    
    # Create a filter for the stats to display in the table
    selected_stats_for_table = st.multiselect(
        "Select metrics to display in the table:",
        options=filtered_table_stats,
        default=filtered_table_stats[:10] if len(filtered_table_stats) > 10 else filtered_table_stats
    )
    
    # Add a checkbox to toggle between seeing all stat details or just percentile ranks
    show_all_stat_details = st.checkbox("Show detailed stats (Sum & Avg)", value=False)
    
    if selected_stats_for_table:
        # Create a combined table with all players' stats
        table_data = []
        
        # Get stats organized by category for better visualization
        categorized_table_stats = []
        for category, stats_list in filtered_stat_categories.items():
            category_stats = [stat for stat in stats_list if stat in selected_stats_for_table]
            if category_stats:
                # If category has any selected stats, add them to the list
                # Add a category header row if more than one category is being displayed
                if selected_table_category == "All" or len(selected_categories) > 1:
                    categorized_table_stats.append((category, None))  # Category header
                # Add all stats for this category
                for stat in category_stats:
                    categorized_table_stats.append((category, stat))
        
        # If no categorized stats (unusual case), just use the flat list
        if not categorized_table_stats:
            for stat in selected_stats_for_table:
                categorized_table_stats.append(("General", stat))
        
        # Process stats in category order
        current_category = None
        for category, stat in categorized_table_stats:
            # If this is a category header row
            if stat is None:
                # Add a category header row
                category_row = {
                    "Metric": f"--- {category} Statistics ---",
                    "_category": category  # For styling
                }
                for name in selected_players:
                    # Add empty cells for each player column
                    if show_all_stat_details:
                        category_row[f"{name} (Rank)"] = ""
                        category_row[f"{name} (Sum)"] = ""
                        category_row[f"{name} (Avg)"] = ""
                    else:
                        category_row[f"{name} (Rank)"] = ""
                
                table_data.append(category_row)
                current_category = category
                continue
            
            # Normal stat row
            stat_row = {
                "Metric": stat,
                "_category": category  # Store category for styling
            }
            
            # Get stats for each player
            for i, (name, percentile_df, actual_df) in enumerate(zip(selected_players, player_percentiles, player_actual_values)):
                if stat in percentile_df.columns and stat in actual_df.columns:
                    percentile = float(percentile_df[stat].iloc[0]) if not pd.isna(percentile_df[stat].iloc[0]) else 0
                    actual_val = float(actual_df[stat].iloc[0]) if not pd.isna(actual_df[stat].iloc[0]) else 0
                    
                    # Get total matches for this player
                    matches = player_info[i].get('total_matches', 1)
                    
                    # Calculate sum differently for special stats
                    if stat in ["Yellow card", "Red card"]:
                        # For cards, count the actual occurrences of non-zero values in the raw data
                        # Get the raw player dataframe
                        raw_df = raw_player_dfs.get(name)
                        if raw_df is not None and stat in raw_df.columns:
                            # Count matches with at least one card (value > 0)
                            card_count = 0
                            for _, match_row in raw_df.iterrows():
                                card_val = pd.to_numeric(match_row[stat], errors='coerce')
                                if pd.notna(card_val) and card_val > 0:
                                    card_count += 1
                            sum_val = card_count
                        else:
                            # Fallback if we can't access raw data
                            sum_val = 0
                    else:
                        # Regular calculation for other stats
                        if matches > 0:
                            # If per90 mode is enabled, the actual_val is already normalized
                            # We just need to calculate the total based on normalized values
                            if use_per90:
                                # For per90 stats, we show the normalized average directly and 
                                # calculate sum by multiplying by matches
                                sum_val = round(actual_val * matches, 2)
                            else:
                                sum_val = round(actual_val * matches, 2)
                        else:
                            sum_val = actual_val
                    
                    # Format and add to the row
                    stat_row[f"{name} (Rank)"] = f"{int(percentile)}%"
                    
                    # Only add sum and avg if detailed view is enabled
                    if show_all_stat_details:
                        # For cards, show count differently
                        if stat in ["Yellow card", "Red card"]:
                            # For cards, display per90 value if per90 mode is enabled
                            if use_per90:
                                stat_row[f"{name} (Sum)"] = f"{sum_val:.2f}"
                                stat_row[f"{name} (Avg)"] = f"{round(actual_val, 2)}/90"
                            else:
                                # For non-per90 mode, show as integer count 
                                card_count = int(round(sum_val))
                                stat_row[f"{name} (Sum)"] = f"{card_count}" if card_count > 0 else "0"
                                stat_row[f"{name} (Avg)"] = f"{round(actual_val, 3):.3f}" if actual_val > 0 else "0"
                        else:
                            # Add per90 label for normalized stats
                            if use_per90 and stat != "Minutes played":
                                stat_row[f"{name} (Sum)"] = f"{sum_val}"
                                avg_label = f"{round(actual_val, 2)}/90"
                            else:
                                stat_row[f"{name} (Sum)"] = f"{sum_val}"
                                avg_label = f"{round(actual_val, 2)}"
                            
                            stat_row[f"{name} (Avg)"] = avg_label
                else:
                    stat_row[f"{name} (Rank)"] = "N/A"
                    # Only add sum and avg if detailed view is enabled
                    if show_all_stat_details:
                        stat_row[f"{name} (Sum)"] = "N/A"
                        stat_row[f"{name} (Avg)"] = "N/A"
            
            table_data.append(stat_row)
        
        # Convert to DataFrame for display
        table_df = pd.DataFrame(table_data)
        
        # Style the DataFrame
        def highlight_cells(val, stat_name=None):
            # List of negative stats where lower values are better
            negative_stats = ["Losses", "Losses own half", "Yellow card", "Red card","Conceded goals","xCG"]
            
            if isinstance(val, str) and val.endswith('%'):
                try:
                    percentile = int(val.rstrip('%'))
                    
                    # For negative stats, invert the color logic
                    if stat_name in negative_stats:
                        percentile = 100 - percentile
                        
                    if percentile > 80:
                        return 'background-color: #1a9641; color: white'
                    elif percentile > 60:
                        return 'background-color: #73c378; color: black'
                    elif percentile > 40:
                        return 'background-color: #f9d057; color: black'
                    elif percentile > 20:
                        return 'background-color: #fc8d59; color: black'
                    else:
                        return 'background-color: #d73027; color: white'
                except:
                    return ''
            return ''
        
        # Apply styling to cells ending with "(Rank)" with proper handling of stat names
        rank_columns = [col for col in table_df.columns if "(Rank)" in col]
        
        # Create a styled table with both cell and row styling
        styled_table = table_df.style
        
        # Apply styling for each cell with the stat name context
        for stat_row in table_data:
            stat_name = stat_row.get("Metric")
            for col in rank_columns:
                # Apply the highlight function with the stat name as context
                styled_table = styled_table.applymap(
                    lambda val, s=stat_name: highlight_cells(val, s), 
                    subset=pd.IndexSlice[table_df[table_df["Metric"] == stat_name].index, col]
                )
        
        # Define category styling
        def category_style(row):
            if "--- Statistics ---" in str(row["Metric"]):
                return ['background-color: #f0f0f0; font-weight: bold; text-align: center; color: #333;'] * len(row)
            
            # Add category-specific styling based on _category column
            category = row.get("_category", "")
            
            if category == "General":
                return ['border-left: 4px solid #1ABC9C;' if col == "Metric" else '' for col in row.index]
            elif category == "Defensive":
                return ['border-left: 4px solid #2E86C1;' if col == "Metric" else '' for col in row.index]
            elif category == "Progressive":
                return ['border-left: 4px solid #8E44AD;' if col == "Metric" else '' for col in row.index]
            elif category == "Offensive":
                return ['border-left: 4px solid #D35400;' if col == "Metric" else '' for col in row.index]
            elif category == "Goalkeeping":
                return ['border-left: 4px solid #9B59B6;' if col == "Metric" else '' for col in row.index]
            elif category == "Distribution":
                return ['border-left: 4px solid #2ECC71;' if col == "Metric" else '' for col in row.index]
            
            return [''] * len(row)
        
        # Apply row-level styling for categories
        if "_category" in table_df.columns:
            styled_table = styled_table.apply(category_style, axis=1)
            # Drop the helper column used for styling before display
            table_df = table_df.drop(columns=["_category"])
        
        # Add a description of the table contents
        st.markdown(f"""
        <div style="margin-bottom: 10px; font-size: 14px;">
          <p>Table showing stats as rows with players in columns:</p>
          <ul style="margin-top: 5px; margin-bottom: 10px;">
            <li><strong>Rank</strong> - Percentile rank (0-100%) showing how the player compares to others</li>
            {"<li><strong>Sum</strong> - Total accumulated statistic across all matches</li>" if show_all_stat_details else ""}
            {"<li><strong>Avg</strong> - " + ("Average value per 90 minutes" if use_per90 else "Average value per match") + "</li>" if show_all_stat_details else ""}
          </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Display the table with custom styling
        st.markdown("""
        <div style="margin-bottom: 8px; font-style: italic; font-size: 12px; color: #666;">
            Click on column headers to sort the table. Scroll horizontally to see all metrics.
        </div>
        """, unsafe_allow_html=True)
        
        # Add additional styling to make the table more readable
        styled_table = styled_table.set_properties(**{
            'font-family': 'Arial, sans-serif',
            'text-align': 'center'
        })
        
        # Change the Metric column to left align
        styled_table = styled_table.set_properties(
            subset=['Metric'], **{'text-align': 'left', 'font-weight': 'bold'}
        )
        
        # Format the DataFrame for better display
        styled_table = styled_table.set_table_styles([
            {'selector': 'th', 'props': [
                ('text-align', 'center'),
                ('background-color', '#f0f0f0'),
                ('font-size', '13px'),
                ('border-bottom', '1px solid #ddd')
            ]}
        ])
        
        # Group player columns visually
        # Get column indices for each player
        for player in selected_players:
            player_cols = [col for col in table_df.columns if player in col]
            if player_cols:
                styled_table = styled_table.set_table_styles([
                    {'selector': f'td:nth-child({table_df.columns.get_loc(player_cols[0]) + 2})', 
                     'props': [('border-left', '2px solid #ddd')]}
                ], overwrite=False)
        
        # Display the table with ability to sort
        st.dataframe(styled_table, use_container_width=True, height=400)
        
        # Add download button for the table
        csv = table_df.to_csv(index=False)
        
    else:
        st.info("Please select at least one metric to display in the table")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add a separator
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Add Forward Type Classification Scatter Plot
    st.markdown('<div class="stats-table-container" style="margin-top: 30px;">', unsafe_allow_html=True)
    
    # Add per90 indication if enabled
    if use_per90:
        st.markdown('<p class="stats-table-header"> Player Type Classification (Per 90 Minutes)</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="stats-table-header"> Player Type Classification</p>', unsafe_allow_html=True)
     
    # Create preset combinations for easier analysis
    preset_combinations = {
        # Forward presets
        "Goals vs xG": ("Goals", "xG"),
        "Shots vs Passes": ("Shots", "Passes accurate"),
        "Dribbles vs Assists": ("Dribbles successful", "Assists"),
        "Duels vs Recoveries": ("Duels won", "Recoveries"),
        "Aerial Duels vs Goals": ("Aerial duels won", "Goals"),
        # Goalkeeper presets
        "Saves vs Conceded": ("Saves", "Conceded goals"),
        "Distribution vs Exits": ("Long passes accurate", "Exits"),
        "Shot Stopping vs Sweeping": ("Saves with reflexes", "Recoveries"),
        "Custom Selection": ("custom", "custom")
    }
    
    # Let user select a preset or custom
    selected_preset = st.selectbox(
        "Choose analysis perspective:",
        options=list(preset_combinations.keys()),
        index=0
    )
    
    # Add explanation for the selected preset
    preset_explanations = {
        # Forward explanations
        "Goals vs xG": "This perspective shows finishing efficiency. "
                      "Advanced Forwards exceed xG, Poachers match xG, "
                      "Deep-Lying Forwards create more than score, and Pressing Forwards have lower values.",
        
        "Shots vs Passes": "This highlights attacking approach: shooting vs playmaking. "
                          "Advanced Forwards balance both, Poachers prioritize shooting, "
                          "Deep-Lying Forwards emphasize passing, and Pressing Forwards focus on pressing.",
        
        "Dribbles vs Assists": "This shows creative attacking style. "
                              "Advanced Forwards excel in both, Poachers focus on finishing, "
                              "Deep-Lying Forwards create chances, and Pressing Forwards contribute through work rate.",
        
        "Duels vs Recoveries": "This focuses on defensive contribution. "
                              "Advanced Forwards win duels in attack, Poachers have limited defensive work, "
                              "Deep-Lying Forwards recover balls deeper, and Pressing Forwards excel in both.",
        
        "Aerial Duels vs Goals": "This shows aerial threat and finishing. "
                                "Advanced Forwards are strong in both, Poachers focus on finishing, "
                                "Deep-Lying Forwards create chances, and Pressing Forwards win aerial battles.",
        
        # Goalkeeper explanations
        "Saves vs Conceded": "This perspective shows shot-stopping efficiency. "
                           "Shot Stoppers excel at making saves and preventing goals, "
                           "while Sweeper Keepers may concede more but contribute to build-up play.",
        
        "Distribution vs Exits": "This highlights goalkeeper's role in possession. "
                               "Shot Stoppers focus on safe distribution, "
                               "while Sweeper Keepers excel at both distribution and defensive exits.",
        
        "Shot Stopping vs Sweeping": "This shows goalkeeper's defensive style. "
                                    "Shot Stoppers excel at making saves, "
                                    "while Sweeper Keepers contribute more to defensive actions outside the box."
    }
    
    if selected_preset in preset_explanations:
        st.markdown(f"""
        <div class="explanation-box" style="font-style: italic; color: #555;">
            {preset_explanations[selected_preset]}
        </div>
        """, unsafe_allow_html=True)
    
    # Get the preset values
    preset_x, preset_y = preset_combinations[selected_preset]
    
    # Create columns for selecting stats for each axis
    scatter_cols = st.columns(2)
    
    # Define default stats
    offense_stats = [stat for stat in available_numeric_stats 
                   if stat in ["Goals", "Shots", "Shots On Target", "xG", "Dribbles successful", "Assists", "Aerial duels won", "Duels won"]]
    
    passing_stats = [stat for stat in available_numeric_stats 
                   if stat in ["Passes accurate", "Long passes accurate", "Crosses accurate", "Assists", 
                              "Dribbles", "Recoveries", "Interceptions", "Duels won"]]
    
    # Filter out card statistics from available stats for both axes
    x_filtered_stats = [stat for stat in available_numeric_stats if stat not in ["Yellow card", "Red card"]]
    y_filtered_stats = [stat for stat in available_numeric_stats if stat not in ["Yellow card", "Red card"]]
    
    default_x_stat = "Goals" if "Goals" in offense_stats else offense_stats[0] if offense_stats else available_numeric_stats[0]
    default_y_stat = "Passes accurate" if "Passes accurate" in passing_stats else passing_stats[0] if passing_stats else available_numeric_stats[0]
    
    with scatter_cols[0]:
        # Create a filter for the X-axis stat
        if preset_x == "custom":
            # Add category filter for stats
            x_stat_category = st.radio(
                "Filter X-axis stats by category:",
                options=["All"] + list(filtered_stat_categories.keys()),
                horizontal=True,
                key="x_stat_category"
            )
            
            # Filter stats based on selected category
            if x_stat_category == "All":
                x_filtered_stats = [stat for stat in available_numeric_stats if stat not in ["Yellow card", "Red card"]]
            else:
                x_filtered_stats = [stat for stat in available_numeric_stats 
                                  if stat in filtered_stat_categories.get(x_stat_category, []) and stat not in ["Yellow card", "Red card"]]
                
                if not x_filtered_stats:  # Fallback if no stats in category
                    x_filtered_stats = [stat for stat in available_numeric_stats if stat not in ["Yellow card", "Red card"]]
            
            x_stat = st.selectbox(
                "X-Axis Statistic:",
                options=x_filtered_stats,
                index=x_filtered_stats.index(default_x_stat) if default_x_stat in x_filtered_stats else 0,
                key="x_stat_selector"
            )
        else:
            # Use preset value if available, fallback to default otherwise
            x_stat = preset_x if preset_x in available_numeric_stats else default_x_stat
            st.markdown(f"**X-Axis**: {x_stat}")
    
    with scatter_cols[1]:
        # Create a filter for the Y-axis stat
        if preset_y == "custom":
            # Add category filter for stats
            y_stat_category = st.radio(
                "Filter Y-axis stats by category:",
                options=["All"] + list(filtered_stat_categories.keys()),
                horizontal=True,
                key="y_stat_category"
            )
            
            # Filter stats based on selected category
            if y_stat_category == "All":
                y_filtered_stats = [stat for stat in available_numeric_stats if stat not in ["Yellow card", "Red card"]]
            else:
                y_filtered_stats = [stat for stat in available_numeric_stats 
                                  if stat in filtered_stat_categories.get(y_stat_category, []) and stat not in ["Yellow card", "Red card"]]
                
                if not y_filtered_stats:  # Fallback if no stats in category
                    y_filtered_stats = [stat for stat in available_numeric_stats if stat not in ["Yellow card", "Red card"]]
            
            y_stat = st.selectbox(
                "Y-Axis Statistic:",
                options=y_filtered_stats,
                index=y_filtered_stats.index(default_y_stat) if default_y_stat in y_filtered_stats else 0,
                key="y_stat_selector"
            )
        else:
            # Use preset value if available, fallback to default otherwise
            y_stat = preset_y if preset_y in available_numeric_stats else default_y_stat
            st.markdown(f"**Y-Axis**: {y_stat}")
    
    
    # Create function to generate an interactive scatter plot with hover
    def create_interactive_scatter(player_names, player_percentiles, player_actual_values, player_colors, x_stat, y_stat, per90_mode=False):
        # List of negative stats where lower values are better
        negative_stats = ["Losses", "Losses own half","Conceded goals","xCG"]  # Yellow card and Red card removed
        
        # Extract data for the scatter plot
        data = []
        for i, (name, percentile_df, actual_df) in enumerate(zip(player_names, player_percentiles, player_actual_values)):
            if x_stat in percentile_df.columns and y_stat in percentile_df.columns:
                x_val = float(percentile_df[x_stat].iloc[0]) if not pd.isna(percentile_df[x_stat].iloc[0]) else 0
                y_val = float(percentile_df[y_stat].iloc[0]) if not pd.isna(percentile_df[y_stat].iloc[0]) else 0
                
                # Ensure values stay within bounds
                x_val = max(5, min(95, x_val))
                y_val = max(5, min(95, y_val))
                
                # Create hover text with details
                hover_text = f"<b>{name}</b><br>"
                
                # Add stat values and percentiles with proper handling for negative stats
                if x_stat in actual_df.columns:
                    x_actual = float(actual_df[x_stat].iloc[0]) if not pd.isna(actual_df[x_stat].iloc[0]) else 0
                    display_x_val = x_val
                    
                    # Add per90 indicator if enabled
                    x_stat_label = f"{x_stat}" + (" (per 90)" if per90_mode and x_stat != "Minutes played" else "")
                    
                    if x_stat in negative_stats:
                        hover_text += f"{x_stat_label}: {x_actual:.2f} ({x_val:.0f}% - lower is better)<br>"
                    else:
                        hover_text += f"{x_stat_label}: {x_actual:.2f} ({x_val:.0f}%)<br>"
                
                if y_stat in actual_df.columns:
                    y_actual = float(actual_df[y_stat].iloc[0]) if not pd.isna(actual_df[y_stat].iloc[0]) else 0
                    display_y_val = y_val
                    
                    # Add per90 indicator if enabled
                    y_stat_label = f"{y_stat}" + (" (per 90)" if per90_mode and y_stat != "Minutes played" else "")
                    
                    if y_stat in negative_stats:
                        hover_text += f"{y_stat_label}: {y_actual:.2f} ({y_val:.0f}% - lower is better)<br>"
                    else:
                        hover_text += f"{y_stat_label}: {y_actual:.2f} ({y_val:.0f}%)<br>"
                
                # Add additional key stats
                additional_stats = ["Goals", "Assists", "Shots", "Passes accurate", "Duels won"]
                for stat in additional_stats:
                    if stat != x_stat and stat != y_stat and stat in actual_df.columns:
                        stat_val = float(actual_df[stat].iloc[0]) if not pd.isna(actual_df[stat].iloc[0]) else 0
                        # Add per90 indicator if enabled
                        stat_label = f"{stat}" + (" (per 90)" if per90_mode and stat != "Minutes played" else "")
                        hover_text += f"{stat_label}: {stat_val:.2f}<br>"
                
                data.append({
                    'name': name,
                    'x': x_val,
                    'y': y_val,
                    'color': player_colors[i % len(player_colors)],
                    'text': hover_text,
                    'x_is_negative': x_stat in negative_stats,
                    'y_is_negative': y_stat in negative_stats
                })
        
        if not data:
            return None
            
        # Create plotly figure with dark theme
        fig = go.Figure()
        
        # Add quadrant lines
        fig.add_shape(
            type="line", x0=0, y0=50, x1=100, y1=50,
            line=dict(color="#666666", width=1)
        )
        fig.add_shape(
            type="line", x0=50, y0=0, x1=50, y1=100,
            line=dict(color="#666666", width=1)
        )
        
        # Create function to generate quadrant descriptions based on selected stats
        def get_quadrant_descriptions(x_stat, y_stat):
            # Define role descriptions based on stat combinations
            role_descriptions = {
                # Goals related combinations
                "Goals": {
                    "high": "Clinical Finisher: High goal output",
                    "low": "Creative Playmaker: Creates chances"
                },
                "xG": {
                    "high": "Efficient Scorer: Exceeds xG",
                    "low": "Chance Creator: Sets up others"
                },
                # Passing related combinations
                "Passes accurate": {
                    "high": "Playmaker: Excellent passing",
                    "low": "Direct Attacker: Focuses on finishing"
                },
                "Assists": {
                    "high": "Creative Forward: Creates chances",
                    "low": "Pure Finisher: Focuses on scoring"
                },
                # Dribbling related combinations
                "Dribbles successful": {
                    "high": "Skilled Dribbler: Takes on defenders",
                    "low": "Positional Player: Relies on movement"
                },
                # Defensive related combinations
                "Duels won": {
                    "high": "Physical Forward: Wins battles",
                    "low": "Technical Forward: Avoids physical play"
                },
                "Recoveries": {
                    "high": "Pressing Forward: High work rate",
                    "low": "Poacher: Minimal defensive work"
                },
                # Aerial related combinations
                "Aerial duels won": {
                    "high": "Aerial Threat: Strong in air",
                    "low": "Technical Player: Ground-based play"
                },
                # Negative stats descriptions
                "Losses": {
                    "high": "Risky: Loses possession frequently",
                    "low": "Safe: Maintains possession well"
                },
                "Losses own half": {
                    "high": "Vulnerable: Loses ball in dangerous areas",
                    "low": "Secure: Protects possession in own half"
                }
            }
            
            # Check if stats are negative (where lower is better)
            x_is_negative = x_stat in ["Losses", "Losses own half","Conceded goals","xCG"]
            y_is_negative = y_stat in ["Losses", "Losses own half","Conceded goals","xCG"]
            
            # Get descriptions for each stat accounting for negative stats
            if x_is_negative:
                # For negative stats, low value is "high" quality and high value is "low" quality
                x_high = role_descriptions.get(x_stat, {}).get("low", "Low " + x_stat + " (good)")
                x_low = role_descriptions.get(x_stat, {}).get("high", "High " + x_stat + " (poor)")
            else:
                x_high = role_descriptions.get(x_stat, {}).get("high", "High " + x_stat)
                x_low = role_descriptions.get(x_stat, {}).get("low", "Low " + x_stat)
                
            if y_is_negative:
                # For negative stats, low value is "high" quality and high value is "low" quality
                y_high = role_descriptions.get(y_stat, {}).get("low", "Low " + y_stat + " (good)")
                y_low = role_descriptions.get(y_stat, {}).get("high", "High " + y_stat + " (poor)")
            else:
                y_high = role_descriptions.get(y_stat, {}).get("high", "High " + y_stat)
                y_low = role_descriptions.get(y_stat, {}).get("low", "Low " + y_stat)
            
            # Generate quadrant descriptions
            return [
                dict(
                    x=25, y=75,
                    text=f"{x_low}<br>{y_high}",
                    showarrow=False,
                    font=dict(color="#AAAAAA", size=12),
                    xanchor="center",
                    yanchor="middle",
                    align="center"
                ),
                dict(
                    x=75, y=75,
                    text=f"{x_high}<br>{y_high}",
                    showarrow=False,
                    font=dict(color="#AAAAAA", size=12),
                    xanchor="center",
                    yanchor="middle",
                    align="center"
                ),
                dict(
                    x=25, y=25,
                    text=f"{x_low}<br>{y_low}",
                    showarrow=False,
                    font=dict(color="#AAAAAA", size=12),
                    xanchor="center",
                    yanchor="middle",
                    align="center"
                ),
                dict(
                    x=75, y=25,
                    text=f"{x_high}<br>{y_low}",
                    showarrow=False,
                    font=dict(color="#AAAAAA", size=12),
                    xanchor="center",
                    yanchor="middle",
                    align="center"
                )
            ]
        
        # Add quadrant descriptions to the plot
        fig.update_layout(
            annotations=get_quadrant_descriptions(x_stat, y_stat)
        )
        
        # Add scatter points for each player
        for player in data:
            fig.add_trace(
                go.Scatter(
                    x=[player['x']],
                    y=[player['y']],
                    mode="markers+text",
                    marker=dict(
                        color=player['color'],
                        size=15,
                        opacity=0.85,
                        line=dict(width=2, color="#222")
                    ),
                    text=player['name'],
                    textposition="bottom center",
                    textfont=dict(
                        color="white",
                        size=15,
                        family="Arial Black, Arial, sans-serif"
                    ),
                    hoverinfo="text",
                    hovertext=player['text'],
                    name=player['name']
                )
            )
        
        # Add title with per90 indication if enabled
        title_text = "Player Type Classification"
        if per90_mode:
            title_text += " (Per 90 Minutes)"
        
        # Configure the layout to match FM24 style
        fig.update_layout(
            plot_bgcolor="#333333",
            paper_bgcolor="#333333",
            width=800,  # Set fixed width
            height=600,  # Set fixed height for better aspect ratio
            xaxis=dict(
                title=dict(text=x_stat.upper() + (" (PER 90)" if per90_mode and x_stat != "Minutes played" else ""), 
                         font=dict(color="#CCCCCC", size=18)),
                range=[0, 100],
                gridcolor="#444444",
                zerolinecolor="#444444",
                tickfont=dict(color="#CCCCCC"),
                showline=True,
                linecolor="#666666",
                tickmode='array',
                tickvals=[0, 25, 50, 75, 100],
                ticktext=['0%', '25%', '50%', '75%', '100%']
            ),
            yaxis=dict(
                title=dict(text=y_stat.upper() + (" (PER 90)" if per90_mode and y_stat != "Minutes played" else ""), 
                         font=dict(color="#CCCCCC", size=18)),
                range=[0, 100],
                gridcolor="#444444",
                zerolinecolor="#444444",
                tickfont=dict(color="#CCCCCC", size=16),
                showline=True,
                linecolor="#666666",
                tickmode='array',
                tickvals=[0, 25, 50, 75, 100],
                ticktext=['0%', '25%', '50%', '75%', '100%']
            ),
            showlegend=False,
            margin=dict(l=60, r=60, t=60, b=60),  # Increased margins for better spacing
            hoverlabel=dict(
                bgcolor="#444444",
                font_size=14,
                font_color="white"
            ),
            # Add title and subtitle
            title=dict(
                text=title_text,
                font=dict(color="#FFFFFF", size=22),
                y=0.95
            ),
        )
        
        return fig
    
    # Add a toggle for interactive mode
    interactive_mode = st.checkbox("Enable Interactive Mode with Hover Details", value=True)
    
    if interactive_mode:
        # Generate and display interactive plotly version
        plotly_fig = create_interactive_scatter(
            selected_players, 
            player_percentiles, 
            player_actual_values, 
            player_colors,
            x_stat,
            y_stat,
            per90_mode=use_per90  # Pass per90 mode flag to the scatter plot function
        )
        
        if plotly_fig:
            st.plotly_chart(plotly_fig, use_container_width=True)
        
        # Add information about negative stats
        if x_stat in ["Losses", "Losses own half","Conceded goals","xCG"] or y_stat in ["Losses", "Losses own half","Conceded goals","xCG"]:
            st.info("""
            **Note about negative statistics:**
            
            For stats like Losses, Losses own half, "Conceded goals",xCG, Yellow card, and Red card, lower values are better.
            These negative stats have been color-coded appropriately:
            - **Red** (0-20%): High frequency (poor performance)
            - **Green** (80-100%): Low frequency (excellent performance)
            
            The percentiles for these stats have been inverted in the visualization so that higher
            percentiles (greener colors) consistently represent better performance.
            """)
        # else:
        #     st.warning("Could not generate interactive scatter plot. Insufficient data.")
    else:
        # Display the static matplotlib version
        scatter_fig = generate_forward_type_scatter(
            selected_players, 
            player_percentiles, 
            player_actual_values, 
            player_colors,
            x_stat,
            y_stat
        )
        st.pyplot(scatter_fig)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add explanatory text about performance metrics (now collapsible)
    with st.expander("About Performance Metrics", expanded=False):
        st.markdown(
            """
**Understanding Performance Metrics**

Each player's performance is measured across various metrics and displayed in both charts and tables:

**Stat Types**
- **Sum**: The total accumulated statistic across all matches (e.g., total goals scored)
- **Average**: """ + ("The average value per 90 minutes of play" if use_per90 else "The average value per match") + """ (e.g., goals per 90 minutes)

**Per 90 Minutes Stats**
""" + ("""
- When "Use Per 90 Minutes Statistics" is enabled, all stats are normalized to a per-90-minutes basis
- This provides a fairer comparison between players with different playing times
- Example: A player with 1 goal in 30 minutes would be credited with 3 goals per 90 minutes
- Statistics like yellow and red cards are also normalized to provide consistent per-90 values
""" if use_per90 else "- Enable the 'Use Per 90 Minutes Statistics' option in the configuration to normalize stats based on playing time") + """

**Percentile Ranking Explained**
- Percentile ranks show how a player compares to others in the current comparison
- A percentile of 75% means the player's value is higher than 75% of the other players in the comparison
- For negative stats (like Losses, Yellow/Red cards), lower values are better, so the percentiles are inverted
- With small datasets (2-3 players), percentiles are adjusted to distribute values more evenly across the 0-100 scale
- The percentileofscore function from SciPy is used to calculate these rankings
- A value of 0 would receive a percentile of 0%, a middle value around 50%, and the highest value 100%

**Color Scale**
- 🟥 **Red** (0-20%): Poor
- 🟧 **Orange** (21-40%): Below Average
- 🟨 **Yellow** (41-60%): Average
- 🟩 **Light Green** (61-80%): Good
- 🟩 **Green** (81-100%): Excellent
            """
        )
    
    # Add a better-styled section for downloading the processed data
    # st.markdown("<div class='download-section'>", unsafe_allow_html=True)
    # st.markdown("<p class='download-title'>Download Processed Data</p>", unsafe_allow_html=True)
    
    # # Create a DataFrame with the percentile ranks for all players
    # download_cols = st.columns(len(selected_players))
    
    # for i, (name, percentile_df, actual_values, col) in enumerate(zip(selected_players, player_percentiles, player_actual_values, download_cols)):
    #     # Convert the first row to CSV for download
    #     if not percentile_df.empty:
    #         csv = percentile_df.to_csv(index=False)
    #         with col:
    #             st.download_button(
    #                 label=f"Download {name} data",
    #                 data=csv,
    #                 file_name=f"{name}_percentiles.csv",
    #                 mime="text/csv"
    #             )
    
    # st.markdown("</div>", unsafe_allow_html=True)

    # Add a separator
    # st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Add DeepSeek AI Insights Section
    st.markdown('<div class="stats-table-container" style="margin-top: 1px;">', unsafe_allow_html=True)
    st.markdown('<p class="stats-table-header">🧠 AI Assistant Insights Analysis</p>', unsafe_allow_html=True)
     
    # Create a trigger button for generating insights
    if st.button("📊 Ask Assistant Insights", use_container_width=True):
        with st.spinner("DeepSeek is analyzing player data..."):
            try:
                # Prepare data for AI analysis
                player_data = []
                for i, (name, percentile_df, actual_df) in enumerate(zip(selected_players, player_percentiles, player_actual_values)):
                    # Get role scores for this player
                    player_role_scores = role_scores[i] if i < len(role_scores) else {}
                    
                    # Get top stats (80th percentile or higher)
                    top_stats = []
                    for stat in percentile_df.columns:
                        percentile = float(percentile_df[stat].iloc[0]) if not pd.isna(percentile_df[stat].iloc[0]) else 0
                        if percentile >= 80:
                            top_stats.append(stat)
                    
                    # Get weak stats (20th percentile or lower)
                    weak_stats = []
                    for stat in percentile_df.columns:
                        percentile = float(percentile_df[stat].iloc[0]) if not pd.isna(percentile_df[stat].iloc[0]) else 0
                        if percentile <= 20:
                            weak_stats.append(stat)
                    
                    # Determine best role based on highest score
                    best_role = max(player_role_scores.items(), key=lambda x: x[1])[0] if player_role_scores else ""
                    
                    # Get position and other key info
                    position = player_info[i].get('position', 'Unknown')
                    matches = player_info[i].get('total_matches', 0)
                    minutes = player_info[i].get('total_minutes', 0)
                    
                    player_data.append({
                        "name": name,
                        "position": position,
                        "matches": matches,
                        "minutes": minutes,
                        "best_role": best_role,
                        "role_scores": player_role_scores,
                        "top_stats": top_stats[:5],  # Limit to top 5 for clarity
                        "weak_stats": weak_stats[:5],  # Limit to top 5 for clarity
                        "percentiles": {col: float(percentile_df[col].iloc[0]) for col in percentile_df.columns if not pd.isna(percentile_df[col].iloc[0])},
                        "actual_values": {col: float(actual_df[col].iloc[0]) for col in actual_df.columns if not pd.isna(actual_df[col].iloc[0])}
                    })
                
                # Get the scatter plot data for default preset
                default_preset = "Goals vs xG"
                x_stat, y_stat = preset_combinations[default_preset]
                
                scatter_data = []
                for i, (name, percentile_df, actual_df) in enumerate(zip(selected_players, player_percentiles, player_actual_values)):
                    if x_stat in percentile_df.columns and y_stat in percentile_df.columns:
                        x_val = float(percentile_df[x_stat].iloc[0]) if not pd.isna(percentile_df[x_stat].iloc[0]) else 0
                        y_val = float(percentile_df[y_stat].iloc[0]) if not pd.isna(percentile_df[y_stat].iloc[0]) else 0
                        
                        # Get actual values
                        x_actual = float(actual_df[x_stat].iloc[0]) if not pd.isna(actual_df[x_stat].iloc[0]) else 0
                        y_actual = float(actual_df[y_stat].iloc[0]) if not pd.isna(actual_df[y_stat].iloc[0]) else 0
                        
                        scatter_data.append({
                            "name": name,
                            "x_stat": x_stat,
                            "y_stat": y_stat,
                            "x_percentile": x_val,
                            "y_percentile": y_val,
                            "x_actual": x_actual,
                            "y_actual": y_actual
                        })
                
                # Use DeepSeek API if API key is provided, otherwise fall back to simulated analysis
                use_api = False
                api_key = st.session_state.get('deepseek_api_key', '')
                
                # Real API implementation
                if api_key:
                    try:
                        import requests
                        import json
                        
                        # Prepare prompt for DeepSeek API
                        prompt = f"""
                        You are an expert football scout and analyst. Please analyze the following player data and provide detailed insights.
                        
                        # Player Data
                        {json.dumps(player_data, indent=2)}
                        
                        # Scatter Plot Data (Goals vs xG)
                        {json.dumps(scatter_data, indent=2)}
                        
                        Please provide a comprehensive analysis covering:
                        1. Performance Overview: Analyze percentile rankings and identify strengths/weaknesses
                        2. Forward Role Assessment: Evaluate suitability for different forward roles
                        3. Positional Analysis: Analyze Goals vs xG data and classify player types
                        4. Development Recommendations: Suggest specific training focus areas
                        
                        Format your response in Markdown with clear headings and sections.
                        """
                        
                        # Call DeepSeek API
                        logger.info("Calling DeepSeek API for player analysis")
                        response = requests.post(
                            "https://api.deepseek.com/v1/chat/completions",
                            headers={
                                "Authorization": f"Bearer {api_key}",
                                "Content-Type": "application/json"
                            },
                            json={
                                "model": "deepseek-chat",
                                "messages": [
                                    {"role": "system", "content": "You are an expert football player analyst and scout."},
                                    {"role": "user", "content": prompt}
                                ],
                                "temperature": 0.7,
                                "max_tokens": 4000
                            }
                        )
                        
                        # Process API response
                        if response.status_code == 200:
                            api_response = response.json()
                            full_analysis = api_response['choices'][0]['message']['content']
                            use_api = False
                            
                            # Display entire response in a nicely formatted container
                            st.markdown("## Complete Analysis from DeepSeek API")
                            st.markdown(full_analysis)
                            
                            # Set flag that insights have been generated
                            st.session_state['insight_generated'] = True
                            
                            # Add download option for API response
                            st.download_button(
                                label="📥 Download Complete API Analysis",
                                data=full_analysis,
                                file_name=f"deepseek_player_insights_{'-'.join(selected_players)}.md",
                                mime="text/markdown"
                            )
                        else:
                            st.warning(f"API request failed with status code {response.status_code}. Falling back to simulated analysis.")
                            logger.warning(f"DeepSeek API request failed: {response.text}")
                    except Exception as api_error:
                        st.warning(f"Error using DeepSeek API: {str(api_error)}. Falling back to simulated analysis.")
                        logger.error(f"Error calling DeepSeek API: {str(api_error)}", exc_info=True)
                
                # If API not used or failed, use simulated analysis
                if not use_api:
                    # Performance Overview Analysis
                    performance_overview = ""
                    for p in player_data:
                        perf_summary = f"### {p['name']} ({p['position']})\n\n"
                        
                        # Overall percentile average
                        avg_percentile = sum(p['percentiles'].values()) / len(p['percentiles']) if p['percentiles'] else 0
                        
                        # Categorize performance
                        if avg_percentile >= 80:
                            performance_tier = "elite"
                        elif avg_percentile >= 65:
                            performance_tier = "above average"
                        elif avg_percentile >= 45:
                            performance_tier = "average"
                        elif avg_percentile >= 30:
                            performance_tier = "below average"
                        else:
                            performance_tier = "needs improvement"
                        
                        perf_summary += f"Overall performance is **{performance_tier}** with an average percentile rank of {avg_percentile:.1f}%. "
                        
                        # Add top strengths
                        if p['top_stats']:
                            perf_summary += f"Key strengths include {', '.join(p['top_stats'])}. "
                        
                        # Add areas for improvement
                        if p['weak_stats']:
                            perf_summary += f"Areas for improvement include {', '.join(p['weak_stats'])}."
                        
                        performance_overview += perf_summary + "\n\n"
                    
                    # Forward Role Assessment
                    role_assessment = ""
                    for p in player_data:
                        # Sort roles by score
                        sorted_roles = sorted(p['role_scores'].items(), key=lambda x: x[1], reverse=True)
                        
                        role_summary = f"### {p['name']} - Role Suitability\n\n"
                        
                        # Add primary role assessment
                        if sorted_roles:
                            primary_role = sorted_roles[0][0]
                            primary_score = sorted_roles[0][1]
                            
                            role_summary += f"**Primary Role: {primary_role}** (Score: {primary_score:.2f})\n\n"
                            
                            # Add explanation based on role
                            if primary_role == "Advance Forward":
                                role_summary += "Player demonstrates excellent goal-scoring ability combined with strong overall attacking presence. "
                                role_summary += "Should be positioned as the main offensive threat with freedom to attack the goal directly.\n\n"
                            elif primary_role == "Pressing Forward":
                                role_summary += "Player excels at defensive contribution and ball recovery in advanced areas. "
                                role_summary += "Should be utilized in a high-pressing system where their work rate disrupts opponent buildup.\n\n"
                            elif primary_role == "Deep-lying Forward":
                                role_summary += "Player shows strong creative passing ability and chance creation. "
                                role_summary += "Best utilized in a withdrawn role where they can link play and create opportunities for teammates.\n\n"
                            elif primary_role == "Poacher":
                                role_summary += "Player demonstrates clinical finishing and efficiency in the box. "
                                role_summary += "Should be positioned to maximize scoring opportunities with minimal defensive responsibility.\n\n"
                            elif primary_role == "Shot Stopper":
                                role_summary += "Player excels at making saves and preventing goals. "
                                role_summary += "Best utilized in a system that requires strong shot-stopping ability and reflexes.\n\n"
                            elif primary_role == "Sweeper Keeper":
                                role_summary += "Player shows excellent ability to read the game and act as an extra defender. "
                                role_summary += "Should be utilized in a high defensive line where their ability to sweep up behind the defense is crucial.\n\n"
                            elif primary_role == "Ball Playing Keeper":
                                role_summary += "Player demonstrates strong distribution and passing ability. "
                                role_summary += "Best utilized in a possession-based system where their ability to start attacks from the back is valuable.\n\n"
                            
                        # Add secondary role if score is within 80% of primary role
                        if len(sorted_roles) > 1:
                            secondary_role = sorted_roles[1][0]
                            secondary_score = sorted_roles[1][1]
                            
                            if secondary_score > 0 and (primary_score == 0 or secondary_score / primary_score >= 0.8):
                                role_summary += f"**Secondary Role: {secondary_role}** (Score: {secondary_score:.2f})\n\n"
                            
                        role_assessment += role_summary + "\n"
                    
                    # Scatter Plot Analysis
                    scatter_analysis = "### Position Analysis (Goals vs xG)\n\n"
                    
                    for p in scatter_data:
                        # Quadrant determination
                        x_high = p['x_percentile'] > 50
                        y_high = p['y_percentile'] > 50
                        
                        # Player type based on quadrant
                        player_type = ""
                        analysis = ""
                        
                        if x_high and y_high:
                            player_type = "Clinical Finisher"
                            analysis = f"**{p['name']}** demonstrates excellent goal scoring ability with high xG production. "
                            if p['x_percentile'] > p['y_percentile']:
                                analysis += f"Outperforming expected goals ({p['x_actual']:.2f} goals vs {p['y_actual']:.2f} xG), showing clinical finishing ability."
                            else:
                                analysis += f"Getting into high-quality scoring positions ({p['y_actual']:.2f} xG) and converting well ({p['x_actual']:.2f} goals)."
                        elif not x_high and y_high:
                            player_type = "Underperforming Finisher"
                            analysis = f"**{p['name']}** is getting into good scoring positions ({p['y_actual']:.2f} xG) but not converting efficiently enough ({p['x_actual']:.2f} goals). "
                            analysis += "Finishing training is recommended to improve conversion rate."
                        elif x_high and not y_high:
                            player_type = "Clinical Opportunist"
                            analysis = f"**{p['name']}** is highly efficient, scoring {p['x_actual']:.2f} goals from limited xG opportunities ({p['y_actual']:.2f}). "
                            analysis += "Very clinical finisher making the most of chances created."
                        else:
                            player_type = "Limited Attacking Threat"
                            analysis = f"**{p['name']}** shows limited attacking output with both goals ({p['x_actual']:.2f}) and xG ({p['y_actual']:.2f}) below average. "
                            analysis += "May be contributing in other areas or need tactical adjustments to increase attacking involvement."
                        
                        scatter_analysis += f"**{p['name']} profile: {player_type}**\n\n{analysis}\n\n"
                    
                    # Development Recommendations
                    development_recommendations = "### Areas for Improvement\n\n"
                    
                    for p in player_data:
                        development_text = f"#### {p['name']}\n\n"
                        
                        # Add improvements based on weak stats
                        if p['weak_stats']:
                            development_text += "🔍 **Priority focus areas:**\n\n"
                            for stat in p['weak_stats'][:3]:  # Top 3 weak areas
                                if "Duel" in stat or "Aerial" in stat:
                                    development_text += f"- 💪 **{stat}**: Increase physical training and focus on positioning for duels\n"
                                elif "Pass" in stat or "Cross" in stat:
                                    development_text += f"- 🎯 **{stat}**: Technical drills focusing on passing accuracy and decision-making\n"
                                elif "Shot" in stat or "Goal" in stat:
                                    development_text += f"- ⚽ **{stat}**: Shooting practice and positioning training for better opportunities\n"
                                else:
                                    development_text += f"- 🔄 **{stat}**: Regular training focus to improve this attribute\n"
                        
                        # Add role recommendation
                        best_role = max(p['role_scores'].items(), key=lambda x: x[1])[0] if p['role_scores'] else ""
                        
                        development_text += f"\n📋 **Recommended tactical role:** {best_role}\n\n"
                        
                        development_recommendations += development_text + "\n\n"
                    
                    # Combine all insights into one comprehensive analysis
                    ai_insights = {
                        "Performance Overview": performance_overview,
                        "Forward Role Assessment": role_assessment,
                        "Positional Analysis": scatter_analysis,
                        "Areas for Improvement": development_recommendations
                    }
                    
                    # Display the insights in tabs
                    ai_tabs = st.tabs(list(ai_insights.keys()))
                    
                    for tab, (section, content) in zip(ai_tabs, ai_insights.items()):
                        with tab:
                            st.markdown("""
                            <style>
                            .insight-container {
                                border-left: 4px solid #4b91e3;
                                padding: 10px 15px;
                                border-radius: 5px;
                                margin-bottom: 10px;
                            }
                            </style>
                            <div class="insight-container">
                            """, unsafe_allow_html=True)
                            st.markdown(content)
                            st.markdown("</div>", unsafe_allow_html=True)
                    
                    
            except Exception as e:
                st.error(f"Error generating AI insights: {str(e)}")
                logger.error(f"Error generating AI insights: {str(e)}", exc_info=True)
    else:
        # Show a prompt if insights haven't been generated yet
        if not st.session_state.get('insight_generated', False):
            st.info("Click the 'Ask Assistant Insights' button above to generate insights about the selected players.")
    
    # This API section is always visible, regardless of insight generation status
    # This section is always visible, whether insights have been generated or not
    st.markdown("---")
    
    # Always show API configuration with visual indicator
    api_key = st.session_state.get('deepseek_api_key', '')
    
    # API Status indicator
    if api_key:
        st.success("✅ DeepSeek API is configured and ready to use")
    else:
        st.warning("⚠️ Using built-in analysis (DeepSeek API not configured)")
    
    # API Configuration section - visible regardless of whether insights were generated
    with st.expander("API Configuration", expanded=False):
        st.markdown("### DeepSeek API Setup")
        api_cols = st.columns([3, 1])
        with api_cols[0]:
            new_api_key = st.text_input(
                "DeepSeek API Key",
                type="password",
                help="Enter your DeepSeek API key for more advanced analysis",
                value=api_key
            )
            
            # Only update if different
            if new_api_key != api_key:
                if new_api_key:
                    st.session_state['deepseek_api_key'] = new_api_key
                    st.success("API key saved! Click the insights button above to use it.")
                elif api_key:  # Only show message if there was a previous key
                    if 'deepseek_api_key' in st.session_state:
                        del st.session_state['deepseek_api_key']
                    st.warning("API key removed. Will use built-in analysis.")
        
        with api_cols[1]:
            if st.button("Clear API Key"):
                if 'deepseek_api_key' in st.session_state:
                    del st.session_state['deepseek_api_key']
                    st.experimental_rerun()

        # Show API benefits
        st.markdown("""
        **Benefits of using the DeepSeek API:**
        - More detailed and nuanced player analysis
        - Advanced natural language understanding of player performance
        - Tactical recommendations based on comprehensive data evaluation
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
     
    with st.expander("What is AI Insights?", expanded=False):
        st.markdown("""
        **DeepSeek AI Insights** analyzes the calculated statistics for selected players and provides:
        
        1. **Performance Overview**: Analysis of percentile rankings across key metrics
        2. **Forward Role Assessment**: Evaluation of each player's suitability for different forward roles
        3. **Player Comparison**: Side-by-side comparison highlighting relative strengths and weaknesses
        4. **Development Recommendations**: Suggestions for areas of improvement based on statistical analysis
        
        This feature uses advanced natural language processing to generate human-readable insights from complex statistical data.
        """)
if __name__ == "__main__":
    main() 