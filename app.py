import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
        st.error("No CSV files found in the data/wyscout/ directory. Please add player CSV files with the naming format 'Player stats <Player Name>.csv'")
        logger.error("No CSV files found in data/wyscout/")
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
                st.success(f"Found player data for {player_name}")
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
            "Visualization type:",
            options=["Bar Chart", "Radar Chart"],
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
                    selected_competitions = st.multiselect(
                        f"Filter by competition:",
                        options=competition_options,
                        default=["All"],
                        key=f"competition_{player}",
                        label_visibility="collapsed"
                    )
                    
                    # If no selections made, default to "All"
                    if not selected_competitions:
                        selected_competitions = ["All"]
                    
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
            "Recoveries opp. half", 
            "Yellow card", 
            "Red card"
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
    
    # Not all columns will be available in the data, so filter to only include those that are
    filtered_categories = {}
    for category, stats in filtered_stat_categories.items():
        available_stats = []
        for stat in stats:
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
    available_numeric_stats = [stat for stat in all_stats if stat in numeric_cols]
    
    # Log the available numeric stats
    logger.info(f"Found {len(available_numeric_stats)} numeric stats for analysis: {', '.join(available_numeric_stats)}")
    
    if not available_numeric_stats:
        st.warning("No common numeric statistics found. Please check your CSV files for numeric metrics.")
        logger.warning("No common numeric statistics found in the data")
        return
    
    # Calculate percentile ranks using only numeric stats
    player_percentiles, player_actual_values = calculate_percentile_ranks(selected_dfs, available_numeric_stats)
    
    if not player_percentiles:
        st.warning("Could not calculate percentile ranks. Please ensure all players have at least some common statistics.")
        return

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
        filtered_table_stats = available_numeric_stats
    else:
        # Get stats from the selected category
        for stat in available_numeric_stats:
            if stat in filtered_stat_categories.get(selected_table_category, []):
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
            
            # Get sum, avg and percentile rank for each player
            for i, (name, percentile_df, actual_df) in enumerate(zip(selected_players, player_percentiles, player_actual_values)):
                if stat in percentile_df.columns and stat in actual_df.columns:
                    percentile = float(percentile_df[stat].iloc[0]) if not pd.isna(percentile_df[stat].iloc[0]) else 0
                    actual_val = float(actual_df[stat].iloc[0]) if not pd.isna(actual_df[stat].iloc[0]) else 0
                    
                    # Calculate sum by multiplying average by number of matches
                    matches = player_info[i].get('total_matches', 1)
                    sum_val = round(actual_val * matches, 2) if matches > 0 else actual_val
                    
                    # Format and add to the row
                    stat_row[f"{name} (Rank)"] = f"{int(percentile)}%"
                    
                    # Only add sum and avg if detailed view is enabled
                    if show_all_stat_details:
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
        def highlight_cells(val):
            if isinstance(val, str) and val.endswith('%'):
                try:
                    percentile = int(val.rstrip('%'))
                    if percentile >= 81:
                        return 'background-color: #4CAF50; color: white'
                    elif percentile >= 61:
                        return 'background-color: #9ACD32; color: black'
                    elif percentile >= 41:
                        return 'background-color: #FFC107; color: black'
                    elif percentile >= 21:
                        return 'background-color: #FF8C00; color: black'
                    else:
                        return 'background-color: #CD5C5C; color: white'
                except:
                    return ''
            return ''
        
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
        
        # Apply styling to cells ending with "(Rank)"
        rank_columns = [col for col in table_df.columns if "(Rank)" in col]
        
        # Create a styled table with both cell and row styling
        styled_table = table_df.style.applymap(highlight_cells, subset=rank_columns)
        
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
        st.download_button(
            label="📊 Download Comparison Table",
            data=csv,
            file_name="player_comparison_table.csv",
            mime="text/csv"
        )
    else:
        st.info("Please select at least one metric to display in the table")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add a separator
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Add Forward Type Classification Scatter Plot
    st.markdown('<div class="stats-table-container" style="margin-top: 30px;">', unsafe_allow_html=True)
    st.markdown('<p class="stats-table-header">Forward Player Type Classification</p>', unsafe_allow_html=True)
    
    # Display information about the scatter plot
    st.markdown("""
    <div class="explanation-box">
        This scatter plot classifies forwards into four types based on their playing style:
        <ul style="margin-top: 5px;">
            <li><strong>Deep-Lying Forward</strong>: Creates chances & links play. Strong at passing & vision.</li>
            <li><strong>Advanced Forward</strong>: All-round attacker. Good at shooting, dribbling & creating.</li>
            <li><strong>Poacher</strong>: Focused on scoring. Excellent positioning & finishing.</li>
            <li><strong>Pressing Forward</strong>: High work rate. Strong at pressing, tackles & winning duels.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Create preset combinations for easier analysis
    preset_combinations = {
        "Goals vs Assists": ("Goals", "Assists"),
        "Shots vs Passes": ("Shots", "Passes accurate"),
        "xG vs Dribbles": ("xG", "Dribbles successful"),
        "Duels won vs Recoveries": ("Duels won", "Recoveries"),
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
        "Goals vs Assists": "This perspective differentiates goal scorers from playmakers. "
                           "Advanced Forwards excel at both, Poachers focus on goals, "
                           "Deep-Lying Forwards create more than score, and Pressing Forwards have moderate values in both.",
        
        "Shots vs Passes": "This highlights the attacking approach: shooting vs passing. "
                          "Advanced Forwards balance both skills, Poachers prioritize shooting over passing, "
                          "Deep-Lying Forwards emphasize passing, and Pressing Forwards contribute with work rate rather than techniques.",
        
        "xG vs Dribbles": "This contrasts finishing quality with dribbling ability. "
                         "Advanced Forwards excel in both areas, Poachers have high xG but fewer dribbles, "
                         "Deep-Lying Forwards may dribble more than score, and Pressing Forwards have moderate values in both.",
        
        "Duels won vs Recoveries": "This focuses on the defensive contributions of forwards. "
                                  "Advanced Forwards win duels in attacking positions, Poachers have limited defensive involvement, "
                                  "Deep-Lying Forwards recover more balls deeper on the pitch, and Pressing Forwards excel in both categories."
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
                x_filtered_stats = available_numeric_stats
            else:
                x_filtered_stats = [stat for stat in available_numeric_stats 
                                  if stat in filtered_stat_categories.get(x_stat_category, [])]
                
                if not x_filtered_stats:  # Fallback if no stats in category
                    x_filtered_stats = available_numeric_stats
            
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
                y_filtered_stats = available_numeric_stats
            else:
                y_filtered_stats = [stat for stat in available_numeric_stats 
                                  if stat in filtered_stat_categories.get(y_stat_category, [])]
                
                if not y_filtered_stats:  # Fallback if no stats in category
                    y_filtered_stats = available_numeric_stats
            
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
    
    # Generate and display the scatter plot
    scatter_fig = generate_forward_type_scatter(
        selected_players, 
        player_percentiles, 
        player_actual_values, 
        player_colors,
        x_stat,
        y_stat
    )
    
    # Wrap the plot in a styled container
    st.markdown('<div class="scatter-plot-container">', unsafe_allow_html=True)
    
    if scatter_fig:
        st.pyplot(scatter_fig)
        
        # Add download button for this specific chart
        st.markdown('<div class="scatter-download-btn">', unsafe_allow_html=True)
        buf = io.BytesIO()
        scatter_fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        
        st.download_button(
            label="📥 Download Scatter Plot",
            data=buf,
            file_name=f"forward_classification_{x_stat}_vs_{y_stat}.png",
            mime="image/png"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Could not generate scatter plot. Insufficient data.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add a separator before the metrics section
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Add explanatory text about performance metrics (now collapsible)
    with st.expander("About Performance Metrics", expanded=False):
        st.markdown("""
        <div style="background-color: #ffffff; border-radius: 8px; padding: 16px; margin: 10px 0;">
            <h3 style="margin-top: 0; color: #333; font-size: 18px;">Understanding Performance Metrics</h3>
            <p style="color: #555; font-size: 14px;">
                Each player's performance is measured across various metrics and displayed in both charts and tables:
            </p>
            
            <h4 style="margin-top: 15px; color: #444; font-size: 16px;">Stat Types</h4>
            <ul style="color: #555; font-size: 14px;">
                <li><strong>Sum</strong> - The total accumulated statistic across all matches (e.g., total goals scored)</li>
                <li><strong>Average</strong> - The average value per match (e.g., average goals per match)</li>
            </ul>
            
            <h4 style="margin-top: 15px; color: #444; font-size: 16px;">Percentile Ranking</h4>
            <p style="color: #555; font-size: 14px;">
                Percentile ranks show how a player compares to others in the comparison. With only 3 players in the dataset, the ranks are calculated using min-max normalization to create a 0-100 scale.
            </p>
            
            <h4 style="margin-top: 15px; color: #444; font-size: 16px;">Color Scale</h4>
            <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px;">
                <div style="display: flex; align-items: center;">
                    <span style="display: inline-block; width: 16px; height: 16px; background-color: #CD5C5C; margin-right: 5px;"></span>
                    <span style="font-size: 14px;"><strong>Red</strong> (0-20%): Poor</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <span style="display: inline-block; width: 16px; height: 16px; background-color: #FF8C00; margin-right: 5px;"></span>
                    <span style="font-size: 14px;"><strong>Orange</strong> (21-40%): Below Average</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <span style="display: inline-block; width: 16px; height: 16px; background-color: #FFC107; margin-right: 5px;"></span>
                    <span style="font-size: 14px;"><strong>Yellow</strong> (41-60%): Average</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <span style="display: inline-block; width: 16px; height: 16px; background-color: #9ACD32; margin-right: 5px;"></span>
                    <span style="font-size: 14px;"><strong>Light Green</strong> (61-80%): Good</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <span style="display: inline-block; width: 16px; height: 16px; background-color: #4CAF50; margin-right: 5px;"></span>
                    <span style="font-size: 14px;"><strong>Green</strong> (81-100%): Excellent</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
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