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
from utils import load_css, get_player_image
from data_utils import extract_player_name, extract_player_info, ensure_numeric_columns, calculate_percentile_ranks
from visualization import generate_unified_player_chart, generate_radar_chart, generate_forward_type_scatter

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
    layout="wide"
)

# Load custom CSS
st.markdown(f'<style>{load_css("styles.css")}</style>', unsafe_allow_html=True)

# Main app
def main():
    st.markdown('<p class="title">Football Player Comparison Tool</p>', unsafe_allow_html=True)
    
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
    
    with st.spinner("Loading player data..."):
        for file in csv_files:
            try:
                player_name = extract_player_name(file)
                player_names.append(player_name)
                player_files[player_name] = file
                
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
        
        # Allow user to select players to compare (multi-select with default to all if ≤ 3)
        default_selections = player_names[:3] if len(player_names) <= 3 else []
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
        available_categories = ["Defensive", "Progressive", "Offensive"]
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
                        
                        df['player_name'] = player  # Add player name to dataframe
                        selected_dfs.append(df)
                else:
                    st.text("No competition data available")
                    
                    # Use the full dataset if no competitions found
                    if player in raw_player_dfs:
                        df = raw_player_dfs[player].copy()
                        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                        df[numeric_cols] = df[numeric_cols].fillna(0)
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

    # Normalize and score each player for each role
    def compute_role_scores(player_actual_values, available_stats, weights):
        # Gather all unique stats from all roles
        all_stats = set()
        for role_weights in weights.values():
            all_stats.update(role_weights.keys())
        # Normalize stats (min-max across all players for each stat)
        stat_min = {stat: min([float(df[stat].iloc[0]) if stat in df.columns else 0 for df in player_actual_values]) for stat in all_stats}
        stat_max = {stat: max([float(df[stat].iloc[0]) if stat in df.columns else 0 for df in player_actual_values]) for stat in all_stats}
        scores = []
        for df in player_actual_values:
            player_score = {}
            for role, role_weights in weights.items():
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
    role_scores = compute_role_scores(player_actual_values, available_numeric_stats, forward_role_weights)
    role_names = list(forward_role_weights.keys())
    
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
            sorted_roles = sorted([(role, player_score[role]) for role in role_names], key=lambda x: x[1], reverse=True)
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

    st.info(
        """
Each player is scored for four classic forward roles based on their stats and the latest competition filter:

- **Advance Forward**: Direct goal threat, excels at finishing and movement.
- **Pressing Forward**: High work rate, presses defenders, wins duels.
- **Deep-lying Forward**: Drops deep, creates chances, links play.
- **Poacher**: Focuses on scoring, operates in the box, exploits chances.
        """
    )

    # Add new table view for player stats comparison
    st.markdown('<div class="stats-table-container">', unsafe_allow_html=True)
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
    show_all_stat_details = st.checkbox("Show detailed stats (Sum & Avg)", value=True)
    
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
                        sum_val = round(actual_val * matches, 2) if matches > 0 else actual_val
                    
                    # Format and add to the row
                    stat_row[f"{name} (Rank)"] = f"{int(percentile)}%"
                    
                    # Only add sum and avg if detailed view is enabled
                    if show_all_stat_details:
                        # For cards, show count differently
                        if stat in ["Yellow card", "Red card"]:
                            # For cards, actual_val is already frequency per match
                            # Convert to count of matches with cards
                            card_count = int(round(sum_val))
                            stat_row[f"{name} (Sum)"] = f"{card_count}" if card_count > 0 else "0"
                            # For average, show the frequency directly
                            stat_row[f"{name} (Avg)"] = f"{round(actual_val, 3):.3f}" if actual_val > 0 else "0"
                        else:
                            stat_row[f"{name} (Sum)"] = f"{sum_val}"
                            stat_row[f"{name} (Avg)"] = f"{round(actual_val, 2)}"
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
            negative_stats = ["Losses", "Losses own half"]  # Yellow card and Red card removed
            
            if isinstance(val, str) and val.endswith('%'):
                try:
                    percentile = int(val.rstrip('%'))
                    
                    # For negative stats, invert the color logic
                    if stat_name in negative_stats:
                        percentile = 100 - percentile
                        
                    if percentile >= 81:
                        return 'background-color: #1a9641; color: white'
                    elif percentile >= 61:
                        return 'background-color: #73c378; color: black'
                    elif percentile >= 41:
                        return 'background-color: #f9d057; color: black'
                    elif percentile >= 21:
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
            {"<li><strong>Avg</strong> - Average value per match</li>" if show_all_stat_details else ""}
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
    st.markdown('<p class="stats-table-header">Forward Player Type Classification</p>', unsafe_allow_html=True)
     
    # Create preset combinations for easier analysis
    preset_combinations = {
        "Goals vs xG": ("Goals", "xG"),
        "Shots vs Passes": ("Shots", "Passes accurate"),
        "Dribbles vs Assists": ("Dribbles successful", "Assists"),
        "Duels vs Recoveries": ("Duels won", "Recoveries"),
        "Aerial Duels vs Goals": ("Aerial duels won", "Goals"),
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
                                "Deep-Lying Forwards create chances, and Pressing Forwards win aerial battles."
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
    def create_interactive_scatter(player_names, player_percentiles, player_actual_values, player_colors, x_stat, y_stat):
        # List of negative stats where lower values are better
        negative_stats = ["Losses", "Losses own half"]  # Yellow card and Red card removed
        
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
                    
                    if x_stat in negative_stats:
                        hover_text += f"{x_stat}: {x_actual:.2f} ({x_val:.0f}% - lower is better)<br>"
                    else:
                        hover_text += f"{x_stat}: {x_actual:.2f} ({x_val:.0f}%)<br>"
                
                if y_stat in actual_df.columns:
                    y_actual = float(actual_df[y_stat].iloc[0]) if not pd.isna(actual_df[y_stat].iloc[0]) else 0
                    display_y_val = y_val
                    
                    if y_stat in negative_stats:
                        hover_text += f"{y_stat}: {y_actual:.2f} ({y_val:.0f}% - lower is better)<br>"
                    else:
                        hover_text += f"{y_stat}: {y_actual:.2f} ({y_val:.0f}%)<br>"
                
                # Add additional key stats
                additional_stats = ["Goals", "Assists", "Shots", "Passes accurate", "Duels won"]
                for stat in additional_stats:
                    if stat != x_stat and stat != y_stat and stat in actual_df.columns:
                        stat_val = float(actual_df[stat].iloc[0]) if not pd.isna(actual_df[stat].iloc[0]) else 0
                        hover_text += f"{stat}: {stat_val:.2f}<br>"
                
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
            x_is_negative = x_stat in ["Losses", "Losses own half"]
            y_is_negative = y_stat in ["Losses", "Losses own half"]
            
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
        
        # Configure the layout to match FM24 style
        fig.update_layout(
            plot_bgcolor="#333333",
            paper_bgcolor="#333333",
            width=800,  # Set fixed width
            height=600,  # Set fixed height for better aspect ratio
            xaxis=dict(
                title=dict(text=x_stat.upper(), font=dict(color="#CCCCCC", size=18)),
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
                title=dict(text=y_stat.upper(), font=dict(color="#CCCCCC", size=18)),
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
                text="Forward Player Type Classification",
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
            y_stat
        )
            
        if plotly_fig:
            st.plotly_chart(plotly_fig, use_container_width=True)
            
            # Add information about negative stats
            if x_stat in ["Losses", "Losses own half"] or y_stat in ["Losses", "Losses own half"]:
                st.info("""
                **Note about negative statistics:**
                
                For stats like Losses, Losses own half, Yellow card, and Red card, lower values are better.
                These negative stats have been color-coded appropriately:
                - **Red** (0-20%): High frequency (poor performance)
                - **Green** (80-100%): Low frequency (excellent performance)
                
                The percentiles for these stats have been inverted in the visualization so that higher
                percentiles (greener colors) consistently represent better performance.
                """)
        else:
            st.warning("Could not generate interactive scatter plot. Insufficient data.")
    else:
        # Display the static matplotlib version
        # Generate and display the scatter plot
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
    
    # Add a separator before the metrics section
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Add explanatory text about performance metrics (now collapsible)
    with st.expander("About Performance Metrics", expanded=False):
        st.markdown(
            """
**Understanding Performance Metrics**

Each player's performance is measured across various metrics and displayed in both charts and tables:

**Stat Types**
- **Sum**: The total accumulated statistic across all matches (e.g., total goals scored)
- **Average**: The average value per match (e.g., average goals per match)

**Percentile Ranking**
- Percentile ranks show how a player compares to others in the comparison. With only 3 players in the dataset, the ranks are calculated using min-max normalization to create a 0-100 scale.

**Color Scale**
- 🟥 **Red** (0-20%): Poor
- 🟧 **Orange** (21-40%): Below Average
- 🟨 **Yellow** (41-60%): Average
- 🟩 **Light Green** (61-80%): Good
- 🟩 **Green** (81-100%): Excellent
            """
        )
    
    # Add a better-styled section for downloading the processed data
    st.markdown("<div class='download-section'>", unsafe_allow_html=True)
    st.markdown("<p class='download-title'>Download Processed Data</p>", unsafe_allow_html=True)
    
    # Create a DataFrame with the percentile ranks for all players
    download_cols = st.columns(len(selected_players))
    
    for i, (name, percentile_df, actual_values, col) in enumerate(zip(selected_players, player_percentiles, player_actual_values, download_cols)):
        # Convert the first row to CSV for download
        if not percentile_df.empty:
            csv = percentile_df.to_csv(index=False)
            with col:
                st.download_button(
                    label=f"Download {name} data",
                    data=csv,
                    file_name=f"{name}_percentiles.csv",
                    mime="text/csv"
                )
    
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main() 