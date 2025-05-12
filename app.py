import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re
import io
import logging
from matplotlib.patches import Patch, Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import matplotlib.image as mpimg
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
import matplotlib.cm as cm

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

# Set page config
st.set_page_config(
    page_title="Player Comparison Tool",
    page_icon="⚽",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .main {
        padding: 1rem;
        background-color: #F9F7F2;
    }
    .title {
        font-size: 2.8rem;
        font-weight: bold;
        margin-bottom: 2rem;
        color: #333;
        text-align: center;
        font-family: 'Arial', sans-serif;
    }
    .subtitle {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #555;
    }
    .player-card {
        padding: 1rem 1rem 1rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        background-color: #FFFFFF;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        display: flex;
        align-items: center;
        border-left: 4px solid #333;
    }
    .player-image {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 1.5rem;
        border: 2px solid #f0f0f0;
    }
    .player-info-container {
        flex-grow: 1;
    }
    .player-name {
        font-size: 2rem;
        font-weight: bold;
        margin: 0;
        padding: 0;
        color: #222;
        font-family: 'Arial', sans-serif;
    }
    .player-info {
        font-size: 1.3rem;
        color: #666;
        margin: 0.3rem 0 0 0;
        padding: 0;
        font-family: 'Arial', sans-serif;
    }
    .chart-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        margin-top: 2rem;
    }
    .stDownloadButton button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 4px;
        font-weight: bold;
    }
    .stDownloadButton button:hover {
        background-color: #45a049;
    }
    .st-emotion-cache-1y4p8pa {
        padding-top: 2rem;
    }
    .st-emotion-cache-16txtl3 h1 {
        font-weight: 800;
        font-family: 'Arial', sans-serif;
    }
    /* Customize the download section */
    .download-section {
        margin-top: 3rem;
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #FFFFFF;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    .download-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #333;
        font-family: 'Arial', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# Function to extract player information from filename
def extract_player_name(filename):
    match = re.search(r"Player stats (.*?)\.csv", os.path.basename(filename))
    if match:
        return match.group(1)
    return os.path.basename(filename)

# Function to extract player info from dataframe
def extract_player_info(df):
    # Get player position (most common one)
    position = "Unknown"
    if 'Position' in df.columns:
        positions = df['Position'].dropna().astype(str)
        if not positions.empty:
            position = positions.value_counts().index[0]
    
    # Get total match count
    total_matches = len(df)
    
    # Get clubs and seasons from competition column
    clubs = []
    seasons = set()
    competitions = set()
    
    if 'Competition' in df.columns:
        for _, row in df.iterrows():
            if pd.notna(row['Competition']):
                competition = str(row['Competition'])
                comp_parts = competition.split('.')[0].strip() if '.' in competition else competition
                competitions.add(comp_parts)
                
                # Try to extract season from date if available
                if 'Date' in df.columns and pd.notna(row['Date']):
                    date_str = str(row['Date'])
                    if '-' in date_str:
                        # Extract year from date (assuming yyyy-mm-dd or dd-mm-yyyy format)
                        date_parts = date_str.split('-')
                        for part in date_parts:
                            if len(part) == 4 and part.isdigit():
                                year = int(part)
                                if 2000 <= year <= 2030:  # Reasonable year range
                                    seasons.add(str(year))
                    elif '/' in date_str:
                        # For formats like yyyy/yyyy
                        seasons.add(date_str.strip())
                
                if " - " in competition:
                    teams = competition.split(" - ")
                    for team in teams:
                        # Remove score if present (e.g., "Team 2:1")
                        team = re.sub(r'\s+\d+:\d+.*$', '', team.strip())
                        if team not in clubs:
                            clubs.append(team)
    
    # If we couldn't find clubs/seasons from the data, check if there are specific club columns
    if not clubs and 'Club' in df.columns:
        clubs_from_col = df['Club'].dropna().astype(str).unique()
        for club in clubs_from_col:
            if club and club.strip():
                clubs.append(club.strip())
    
    # If we couldn't find seasons from the data, check if there are specific season columns
    if not seasons and 'Season' in df.columns:
        seasons_from_col = df['Season'].dropna().astype(str).unique()
        for season in seasons_from_col:
            if season and season.strip():
                seasons.add(season.strip())
    
    # Get the main club
    club = clubs[0] if clubs else "Unknown"
    
    # Calculate total minutes played
    total_minutes = 0
    if 'Minutes played' in df.columns:
        total_minutes = pd.to_numeric(df['Minutes played'], errors='coerce').fillna(0).sum()
    
    # Calculate total goals
    total_goals = 0
    if 'Goals' in df.columns:
        total_goals = pd.to_numeric(df['Goals'], errors='coerce').fillna(0).sum()
    
    # Calculate total seasons and total competitions
    total_seasons = len(seasons)
    total_competitions = len(competitions)
    
    # Set default club values if none found in data
    club_map = {
        "Gustavo Henrique": "Zhako",
        "Ribamar": "Dhing A Thanh Hoa",
        "Uilliam": "Al-Fahaleel SC"
    }
    
    player_name = df['player_name'].iloc[0] if 'player_name' in df.columns else "Unknown"
    # Use the mapped club if the default "Unknown" was detected
    if club == "Unknown" and player_name in club_map:
        club = club_map[player_name]
    
    # Set a default age based on some common values from the sample
    # In a real app, you would extract this from the data
    age_map = {
        "Gustavo Henrique": 29,
        "Ribamar": 27,
        "Uilliam": 30
    }
    
    age = age_map.get(player_name, 25)  # Default age 25 if not found
    
    # Try to extract age from Age column if it exists
    if 'Age' in df.columns:
        age_values = pd.to_numeric(df['Age'].dropna(), errors='coerce')
        if not age_values.empty and age_values.notna().any():
            try:
                # Get the most common age value
                age = int(age_values.mode().iloc[0])
            except:
                pass
    
    return {
        "position": position,
        "club": club,
        "age": age,
        "total_matches": total_matches,
        "total_minutes": int(total_minutes),
        "total_goals": int(total_goals),
        "total_seasons": total_seasons,
        "total_competitions": total_competitions
    }

# Function to handle cleaning data to ensure columns are numeric
def ensure_numeric_columns(df, exclude_columns=None):
    """
    Ensure that all columns in a dataframe are numeric, excluding specified columns.
    Returns a new dataframe with only the numeric columns.
    """
    if exclude_columns is None:
        exclude_columns = ["Match", "Competition", "Date", "Position"]
    
    # First exclude the non-numeric columns
    cols_to_check = [col for col in df.columns if col not in exclude_columns]
    
    # Create a new dataframe with only convertible numeric columns
    numeric_df = pd.DataFrame(index=df.index)
    
    for col in cols_to_check:
        try:
            # Try to convert to numeric
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            # Only add if not all NaN
            if not numeric_series.isna().all():
                numeric_df[col] = numeric_series
        except Exception as e:
            # Log the error and skip this column
            logger.warning(f"Could not convert column '{col}' to numeric: {str(e)}")
            pass
    
    # If no numeric columns were found, log a warning
    if numeric_df.empty:
        logger.warning("No numeric columns found in the dataframe after filtering")
    
    return numeric_df

# Function to calculate percentile ranks
def calculate_percentile_ranks(df_list, stat_cols):
    # Define columns that should never be treated as numeric
    non_numeric_columns = ["Match", "Competition", "Date", "Position"]
    
    # Remove any non-numeric columns from stat_cols
    numeric_stat_cols = [col for col in stat_cols if col not in non_numeric_columns]
    
    # If no numeric columns are left, return empty
    if not numeric_stat_cols:
        logger.warning("No numeric columns found for percentile calculation")
        return [], []
    
    # Combine all player stats for ranking
    combined_stats = pd.DataFrame()
    
    # Filter to include only numeric columns
    for i, df in enumerate(df_list):
        # First ensure we have numeric data only
        numeric_df = ensure_numeric_columns(df, non_numeric_columns)
        
        # Filter to only include the requested stats
        valid_cols = [col for col in numeric_stat_cols if col in numeric_df.columns]
        
        if valid_cols:
            player_stats = numeric_df[valid_cols].mean().to_frame().T
            player_stats['player_index'] = i
            combined_stats = pd.concat([combined_stats, player_stats], ignore_index=True)
    
    # If no valid numeric columns found, return empty
    if combined_stats.empty:
        logger.warning(f"No valid numeric columns found in the datasets")
        return [], []
    
    # Filter out columns that don't exist in all dataframes or have no variance
    valid_cols = []
    for col in numeric_stat_cols:
        if col in combined_stats.columns and combined_stats[col].nunique() > 1:
            valid_cols.append(col)
    
    if not valid_cols:
        logger.warning(f"No valid columns with variance found")
        return [], []
    
    # Check if we have enough data to rank
    if len(combined_stats) < 2:
        logger.warning(f"Not enough data to calculate percentile ranks")
        return [], []
    
    # Calculate percentile ranks
    percentile_ranks = pd.DataFrame()
    actual_values = pd.DataFrame()  # Store actual values
    
    for col in valid_cols:
        if col in combined_stats.columns:
            # Store the actual values
            actual_values[col] = combined_stats[col]
            
            # Determine if higher is better (default) or lower is better
            higher_is_better = True
            lower_is_better_cols = [
                'Losses', 'Losses own half', 'Yellow card', 'Red card'
            ]
            
            # Check if any of the lower_is_better substrings are in the column name
            for lower_col in lower_is_better_cols:
                if lower_col in col:
                    higher_is_better = False
                    break
            
            # Calculate percentile rank
            if higher_is_better:
                # Create a normalized percentile (0-100 scale) based on min-max
                min_val = combined_stats[col].min()
                max_val = combined_stats[col].max()
                
                if min_val == max_val:  # If all values are the same
                    percentile_ranks[col] = 50  # Assign mid-range value
                else:
                    percentile_ranks[col] = 100 * (combined_stats[col] - min_val) / (max_val - min_val)
            else:
                # For metrics where lower is better, invert the percentile
                min_val = combined_stats[col].min()
                max_val = combined_stats[col].max()
                
                if min_val == max_val:  # If all values are the same
                    percentile_ranks[col] = 50  # Assign mid-range value
                else:
                    percentile_ranks[col] = 100 * (max_val - combined_stats[col]) / (max_val - min_val)
    
    percentile_ranks['player_index'] = combined_stats['player_index']
    actual_values['player_index'] = combined_stats['player_index']
    
    # Split back into individual player percentile ranks and actual values
    player_percentiles = []
    player_actual_values = []
    
    for i in range(len(df_list)):
        player_percentiles.append(
            percentile_ranks[percentile_ranks['player_index'] == i].drop(columns=['player_index'])
        )
        player_actual_values.append(
            actual_values[actual_values['player_index'] == i].drop(columns=['player_index'])
        )
    
    return player_percentiles, player_actual_values

# Function to generate a unified player stat chart
def generate_unified_player_chart(player_name, percentile_df, player_color, player_info, player_image_path=None, actual_values_df=None):
    # Get all stats from the percentile dataframe
    if percentile_df.empty:
        return None
    
    # Create figure
    fig = plt.figure(figsize=(6, 12))
    
    # Create a gridspec layout with space for player info at top
    gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 9], hspace=0.05)
    
    # Add player info at top
    ax_info = fig.add_subplot(gs[0])
    ax_info.axis('off')  # Turn off axis
    ax_info.set_facecolor('#F9F7F2')
    
    # Add player name as title
    ax_info.text(0.02, 0.8, player_name, fontsize=14, fontweight='bold', color="#333333")
    
    # Add player details
    position = player_info.get('position', 'Unknown')
    club = player_info.get('club', 'Unknown')
    age = player_info.get('age', 'Unknown')
    
    # Add basic info
    ax_info.text(0.02, 0.6, f"{age} | {position} | {club}", fontsize=10, color="#555555")
    
    # Add stats info (matches, seasons, clubs)
    total_matches = player_info.get('total_matches', 0)
    total_seasons = player_info.get('total_seasons', 0)
    total_minutes = player_info.get('total_minutes', 0)
    total_goals = player_info.get('total_goals', 0)
    
    stats_info_text = f"Matches: {total_matches} | Seasons: {total_seasons} | Minutes: {total_minutes} | Goals: {total_goals}"
    ax_info.text(0.02, 0.4, stats_info_text, fontsize=9, color="#666666")
    
    # If player image is available, add it
    if player_image_path and os.path.exists(player_image_path):
        try:
            img = mpimg.imread(player_image_path)
            # Add a small inset axes for the image
            ax_img = fig.add_axes([0.7, 0.92, 0.2, 0.2], frameon=True)
            ax_img.imshow(img)
            ax_img.axis('off')
        except Exception as e:
            print(f"Error loading image: {e}")
    
    # Create the main chart
    ax = fig.add_subplot(gs[1])
    ax.set_facecolor('#F9F7F2')
    fig.patch.set_facecolor('#F9F7F2')  # Light cream background
    
    # Get all stats
    all_stats = list(percentile_df.columns)
    
    # Define category for each stat
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
    
    # Add any other stats from the original list
    original_stats = [
        "Match", 
        "Competition", 
        "Date", 
        "Position", 
        "Minutes played", 
        "Total actions", 
        "Total actions successful", 
        "Goals", 
        "Assists", 
        "Shots", 
        "Shots On Target", 
        "xG", 
        "Passes", 
        "Passes accurate", 
        "Long passes", 
        "Long passes accurate", 
        "Crosses", 
        "Crosses accurate", 
        "Dribbles", 
        "Dribbles successful", 
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
    ]
    
    # Map original stats to categories
    for stat in original_stats:
        if stat in ["Match", "Competition", "Date", "Position", "Minutes played", 
                   "Total actions", "Total actions successful"]:
            if stat not in stat_categories["General"]:
                stat_categories["General"].append(stat)
        elif any(defensive_term in stat for defensive_term in 
                ["Duel", "Interception", "Loss", "Recover", "Yellow card", "Red card"]):
            if stat not in stat_categories["Defensive"]:
                stat_categories["Defensive"].append(stat)
        elif any(progressive_term in stat for progressive_term in 
                ["Pass", "Dribble", "Cross"]):
            if stat not in stat_categories["Progressive"]:
                stat_categories["Progressive"].append(stat)
        elif any(offensive_term in stat for offensive_term in 
                ["Goal", "Assist", "Shot", "xG"]):
            if stat not in stat_categories["Offensive"]:
                stat_categories["Offensive"].append(stat)
    
    # Categorize all stats
    categorized_stats = []
    category_dividers = []
    current_pos = 0
    
    for category, category_stats in stat_categories.items():
        category_stats_present = [stat for stat in category_stats if stat in all_stats]
        if category_stats_present:
            categorized_stats.extend(category_stats_present)
            current_pos += len(category_stats_present)
            category_dividers.append((current_pos, category))
    
    # Handle stats that aren't in any category
    general_stats = [stat for stat in all_stats if stat not in categorized_stats]
    if general_stats:
        categorized_stats = general_stats + categorized_stats
        # Shift all category dividers
        category_dividers = [(pos + len(general_stats), cat) for pos, cat in category_dividers]
        # Add general category if needed
        if general_stats:
            category_dividers.insert(0, (len(general_stats), "General"))
    
    # Filter out stats that are in categorized_stats but not in all_stats
    valid_stats = [stat for stat in categorized_stats if stat in all_stats]
    
    # Get percentile values
    percentile_values = []
    for stat in valid_stats:
        if stat in percentile_df.columns:
            val = float(percentile_df[stat].iloc[0]) if not pd.isna(percentile_df[stat].iloc[0]) else 0
            percentile_values.append(val)
        else:
            percentile_values.append(0)
    
    # Get actual values if provided
    actual_values = []
    if actual_values_df is not None:
        for stat in valid_stats:
            if stat in actual_values_df.columns:
                val = float(actual_values_df[stat].iloc[0]) if not pd.isna(actual_values_df[stat].iloc[0]) else 0
                # Round to 2 decimal places for display
                val = round(val, 2)
                actual_values.append(val)
            else:
                actual_values.append(0)
    else:
        actual_values = [0] * len(percentile_values)
    
    # Calculate positions for bars
    y_positions = np.arange(len(valid_stats))
    
    # Generate colors for bars based on percentile values
    bar_colors = []
    for value in percentile_values:
        bar_colors.append(get_percentile_color(value))
    
    # Plot horizontal bars - make them more compact (smaller height)
    bars = ax.barh(
        y_positions,
        percentile_values,
        height=0.4,  # More compact bars
        color=bar_colors,
        alpha=0.9
    )
    
    # Add value labels to bars
    for i, (bar, percentile_val, actual_val) in enumerate(zip(bars, percentile_values, actual_values)):
        if percentile_val > 5:  # Only add text if bar is wide enough
            # Display both the percentile and actual value
            display_text = f"{int(percentile_val)}% | {actual_val}"
            ax.text(
                min(percentile_val - 3, 95),  # Position text inside the bar near the end
                bar.get_y() + bar.get_height()/2,
                display_text,
                va='center',
                ha='right',
                fontsize=7,
                fontweight='bold',
                color='white'
            )
    
    # Set labels and ticks
    ax.set_yticks(y_positions)
    ax.set_yticklabels(valid_stats, fontsize=8)  # Smaller font for more compact view
    ax.set_xlabel('Percentile Rank', fontsize=8)
    
    # Set x-axis range and ticks
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis='x', labelsize=7)  # Smaller tick font size
    
    # Add gridlines
    ax.grid(axis='x', linestyle='--', alpha=0.2, color='gray')
    
    # Add vertical lines at 0, 20, 40, 60, 80, 100
    for x in [0, 20, 40, 60, 80, 100]:
        ax.axvline(x=x, color='gray', linestyle='-', alpha=0.15)
    
    # Add category dividers and labels
    for i, (pos, category) in enumerate(category_dividers):
        if i > 0:  # Skip the first divider
            y_pos = y_positions[pos-1] + 0.5 if pos < len(y_positions) else len(y_positions) - 0.5
            ax.axhline(y=y_pos, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        
        # Calculate middle position for category label
        start_idx = 0 if i == 0 else category_dividers[i-1][0]
        end_idx = pos
        if start_idx < end_idx and end_idx <= len(y_positions):
            mid_pos = (y_positions[start_idx] + y_positions[min(end_idx-1, len(y_positions)-1)]) / 2
            
            # Add category label on the right side
            rect_height = 0.7 * (end_idx - start_idx)
            rect_y = mid_pos - rect_height/2
            
            # Use different colors for different categories
            rect_color = '#2E86C1' if category == 'Defensive' else '#D35400' if category == 'Offensive' else '#8E44AD' if category == 'Progressive' else '#1ABC9C'
            
            # Add colored rectangle on the right
            rect = plt.Rectangle((1.01, rect_y), 0.03, rect_height, 
                                transform=ax.transAxes, color=rect_color, alpha=0.8)
            ax.add_patch(rect)
            
            # Add text - change to black for better readability
            ax.text(1.04, mid_pos, category, transform=ax.get_yaxis_transform(), 
                    rotation=270, fontsize=10, fontweight='bold', 
                    ha='center', va='center', color='black')
    
    # Add horizontal lines at each position for better readability
    for y in y_positions:
        ax.axhline(y=y-0.3, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_color('#555555')
    ax.spines['left'].set_color('#555555')
    
    # Add legend at the bottom
    legend_elements = [
        Patch(facecolor='#CD5C5C', label='0-20'),
        Patch(facecolor='#FF8C00', label='21-40'),
        Patch(facecolor='#FFC107', label='41-60'),
        Patch(facecolor='#9ACD32', label='61-80'),
        Patch(facecolor='#4CAF50', label='81-100')
    ]
    
    # Create a separate axes for the legend at the bottom
    legend_ax = fig.add_axes([0.1, 0.02, 0.8, 0.02], frameon=False)
    legend_ax.axis('off')
    legend = legend_ax.legend(handles=legend_elements, loc='center', 
                             ncol=5, frameon=False, fontsize=7,
                             title="Percentile Rank", title_fontsize=8)
    
    plt.tight_layout()
    # Adjust the main plot to make room for the legend
    plt.subplots_adjust(bottom=0.07)
    
    return fig

# Custom class for Radar chart display
class RadarAxes(PolarAxes):
    name = 'radar'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_theta_zero_location('N')
    
    def fill(self, *args, closed=True, **kwargs):
        return super().fill(closed=closed, *args, **kwargs)
    
    def plot(self, *args, **kwargs):
        lines = super().plot(*args, **kwargs)
        for line in lines:
            self._close_line(line)
        return lines
    
    def _close_line(self, line):
        x, y = line.get_data()
        if x[0] != x[-1]:
            x = np.concatenate((x, [x[0]]))
            y = np.concatenate((y, [y[0]]))
            line.set_data(x, y)

# Register the RadarAxes projection
register_projection(RadarAxes)

# Function to generate radar chart for player comparison
def generate_radar_chart(player_names, player_percentiles, player_colors, player_actual_values=None):
    if not player_names or not player_percentiles:
        logger.warning("No player data provided for radar chart")
        return None
    
    try:
        logger.info(f"Generating radar chart for players: {', '.join(player_names)}")
        
        # Find common stats across all players
        common_stats = set(player_percentiles[0].columns)
        for percentile_df in player_percentiles[1:]:
            common_stats &= set(percentile_df.columns)
        
        logger.info(f"Found {len(common_stats)} common stats across all players")
        
        if not common_stats:
            logger.warning("No common statistics found across selected players")
            # If no common stats found, return a simple figure with an error message
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No common statistics found across selected players.", 
                    ha='center', va='center', fontsize=14, color='red')
            ax.axis('off')
            return fig
        
        # Define key stats to prioritize in radar chart
        key_stats = [
            'Goals', 'Assists', 'Shots On Target',
            'Passes accurate', 'Dribbles successful',
            'Duels won', 'Interceptions',
            'Recoveries', 'xG', 'Minutes played'
        ]
        
        # Filter for key stats that exist in common stats
        selected_stats = [stat for stat in key_stats if stat in common_stats]
        logger.info(f"Found {len(selected_stats)} key stats in common stats")
        
        # If we don't have enough key stats, add more from common_stats
        if len(selected_stats) < 6:
            logger.info(f"Adding more stats to reach minimum of 6 stats")
            other_stats = list(common_stats - set(selected_stats))
            # Sort by name to ensure consistent order
            other_stats.sort()
            # Add more until we have at least 6 or run out
            selected_stats.extend(other_stats[:max(6 - len(selected_stats), 0)])
        
        # Limit to max 12 stats for readability
        if len(selected_stats) > 12:
            logger.info(f"Limiting to max 12 stats from {len(selected_stats)} available")
            selected_stats = selected_stats[:12]
        
        # Ensure we have at least 3 stats for the radar chart to work
        if len(selected_stats) < 3:
            logger.warning(f"Only {len(selected_stats)} stats available - need at least 3 for radar chart")
            # Not enough stats for radar chart, return a simple figure with error
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "Insufficient statistics for radar chart. Need at least 3 common metrics.", 
                    ha='center', va='center', fontsize=14, color='red')
            ax.axis('off')
            return fig
        
        # Set up radar chart using a simpler approach
        N = len(selected_stats)
        logger.info(f"Creating radar chart with {N} axes")
        
        # Calculate angles for each feature
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        # Close the polygon by repeating the first angle
        angles += angles[:1]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'polar': True})
        
        # Background styling
        fig.patch.set_facecolor('#F9F7F2')  # Light cream background
        
        # Create a table to display actual values below the chart
        if player_actual_values:
            actual_data = []
            for i, (name, actual_df) in enumerate(zip(player_names, player_actual_values)):
                row = [name]
                for stat in selected_stats:
                    if stat in actual_df.columns:
                        val = round(float(actual_df[stat].iloc[0]) if not pd.isna(actual_df[stat].iloc[0]) else 0, 2)
                        row.append(val)
                    else:
                        row.append(0)
                actual_data.append(row)
        
        # Loop over each player and plot their data
        for i, (name, percentile_df) in enumerate(zip(player_names, player_percentiles)):
            color = player_colors[i % len(player_colors)]
            
            # Get percentile values for selected stats
            values = []
            for stat in selected_stats:
                if stat in percentile_df.columns:
                    val = float(percentile_df[stat].iloc[0]) if not pd.isna(percentile_df[stat].iloc[0]) else 0
                    # Convert to 0-1 scale for radar chart
                    values.append(val / 100.0)
                else:
                    values.append(0)
            
            # Add the first value again to close the polygon
            values += values[:1]
            
            logger.info(f"Plotting data for player: {name}")
            # Plot the player data
            ax.plot(angles, values, color=color, linewidth=2.5, label=name)
            ax.fill(angles, values, color=color, alpha=0.25)
        
        # Set the angle labels (feature names)
        ax.set_xticks(angles[:-1])  # Exclude the last angle which is a duplicate
        ax.set_xticklabels([stat[:20] + '...' if len(stat) > 20 else stat for stat in selected_stats], 
                          fontsize=9)
        
        # Set y-ticks (concentric circles) to show percentile values
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=8)
        
        # Set the limit of the radar chart
        ax.set_ylim(0, 1)
        
        # Add grid lines
        ax.grid(True, alpha=0.3)
        
        # Add a legend with title
        legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                          ncol=3, fontsize=11, frameon=True, title="Player Comparison")
        legend.get_title().set_fontweight('bold')
        
        # Add title
        plt.figtext(0.5, 0.965, 'Player Comparison: Percentile Ranks', 
                   ha='center', color='#333333', weight='bold', size=16)
        plt.figtext(0.5, 0.93, ', '.join(player_names), 
                   ha='center', color='#666666', size=12)
        
        # If we have actual values, add a table below the chart
        if player_actual_values and 'actual_data' in locals():
            # Add the actual values in a text box below
            table_text = "Actual Values:\n\n"
            
            # Add header row
            header = ["Player"] + [stat[:10] + '...' if len(stat) > 10 else stat for stat in selected_stats]
            table_text += " | ".join(header) + "\n"
            table_text += "-" * len(table_text) + "\n"
            
            # Add data rows
            for row in actual_data:
                table_text += " | ".join([str(x) for x in row]) + "\n"
            
            # Add text box with actual values
            plt.figtext(0.5, -0.15, table_text, 
                      ha='center', fontsize=8, family='monospace',
                      bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
            
            # Adjust the figure to make room for the table
            plt.subplots_adjust(bottom=0.25)
        
        # Adjust layout
        try:
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust to make room for the values table
        except Exception as layout_error:
            logger.warning(f"tight_layout failed: {str(layout_error)}")
            # If tight_layout fails, use a standard adjustment
            plt.subplots_adjust(top=0.9, bottom=0.25)  # More space at bottom for the table
        
        logger.info("Radar chart generation completed successfully")
        return fig
    except Exception as e:
        logger.error(f"Error generating radar chart: {str(e)}", exc_info=True)
        # Create a figure with error message
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"Error generating radar chart: {str(e)}", 
                ha='center', va='center', fontsize=14, color='red')
        ax.axis('off')
        return fig

# Function to get player image
def get_player_image(player_name):
    # Check if there's an image file with the player's name in the data/pics directory
    image_path = f"data/pics/{player_name}.png"
    if os.path.exists(image_path):
        return image_path
    
    # If not, check for other image formats
    for ext in ['.png', '.jpeg', '.gif']:
        alt_path = f"data/pics/{player_name}{ext}"
        if os.path.exists(alt_path):
            return alt_path
    
    # If no specific player image is found, use a default image
    default_image = "data/pics/default_player.jpg"
    if os.path.exists(default_image):
        return default_image
    
    # If no default image either, return None
    return None

# Function to get color based on percentile
def get_percentile_color(value):
    """Return a color based on the percentile value using a gradient scale."""
    if value < 20:
        return '#CD5C5C'  # Red
    elif value < 40:
        return '#FF8C00'  # Dark Orange
    elif value < 60:
        return '#FFC107'  # Amber/Yellow
    elif value < 80:
        return '#9ACD32'  # Light Green (Yellow-Green)
    else:
        return '#4CAF50'  # Green

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
        <div style="background-color: #EBF5FB; border-radius: 8px; padding: 10px; margin-bottom: 10px;">
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
    
    # Add explanatory text about percentile ranks
    st.markdown("""
    <div style="background-color: #f0f4f8; border-radius: 8px; padding: 16px; margin: 20px 0; border-left: 4px solid #3498db;">
        <h3 style="margin-top: 0; color: #333;">About Percentile Ranks</h3>
        <p style="color: #555; font-size: 16px;">
            Each player's chart shows their normalized percentile rank (0-100) across various performance metrics.
            The percentiles are calculated based on the min-max values of each metric across all players.
            <br><br>
            Higher values (closer to 100) indicate better performance relative to other players in the comparison.
            For metrics where lower values are better (e.g., losses, yellow cards), the scale is inverted.
            <br><br>
            <b>Color Scale:</b>
            <br>
            <span style="display: inline-block; width: 12px; height: 12px; background-color: #CD5C5C; margin-right: 5px;"></span> <b>Red</b> (0-20): Poor performance
            <br>
            <span style="display: inline-block; width: 12px; height: 12px; background-color: #FF8C00; margin-right: 5px;"></span> <b>Orange</b> (21-40): Below average performance
            <br>
            <span style="display: inline-block; width: 12px; height: 12px; background-color: #FFC107; margin-right: 5px;"></span> <b>Yellow</b> (41-60): Average performance
            <br>
            <span style="display: inline-block; width: 12px; height: 12px; background-color: #9ACD32; margin-right: 5px;"></span> <b>Light Green</b> (61-80): Good performance
            <br>
            <span style="display: inline-block; width: 12px; height: 12px; background-color: #4CAF50; margin-right: 5px;"></span> <b>Green</b> (81-100): Excellent performance
        </p>
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

# Function to convert image to base64 for HTML embedding
def image_to_base64(image_path):
    import base64
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

if __name__ == "__main__":
    main() 