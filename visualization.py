import matplotlib.pyplot as plt
import numpy as np
import os
import io
import pandas as pd
import logging
import matplotlib.image as mpimg
from utils import get_player_image, get_percentile_color, get_percentile_legend_elements

logger = logging.getLogger("player_comparison_tool")

# Function to generate a unified player stat chart
def generate_unified_player_chart(player_name, percentile_df, player_color, player_info, player_image_path=None, actual_values_df=None):
    """
    Generate a unified player stat chart showing percentile ranks for various metrics.
    
    Args:
        player_name (str): Name of the player
        percentile_df (pandas.DataFrame): DataFrame containing percentile ranks
        player_color (str): Color to use for the player's chart
        player_info (dict): Dictionary containing player information
        player_image_path (str, optional): Path to player image. Defaults to None.
        actual_values_df (pandas.DataFrame, optional): DataFrame containing actual values. Defaults to None.
        
    Returns:
        matplotlib.figure.Figure: The generated chart figure
    """
    # Get all stats from the percentile dataframe
    if percentile_df.empty:
        return None
    
    # Create figure with improved proportions
    fig = plt.figure(figsize=(6, 10))
    
    # Create a gridspec layout with space for player info at top
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 8], hspace=0.05)
    
    # Add player info at top with improved styling
    ax_info = fig.add_subplot(gs[0])
    ax_info.axis('off')  # Turn off axis
    ax_info.set_facecolor('#F9F7F2')
    
    # Add player name as title with more prominent styling
    ax_info.text(0.02, 0.8, player_name, fontsize=16, fontweight='bold', color="#333333")
    
    # Add player details with better formatting and spacing
    position = player_info.get('position', 'Unknown')
    club = player_info.get('club', 'Unknown')
    age = player_info.get('age', 'Unknown')
    
    # Add basic info with cleaner typography
    ax_info.text(0.02, 0.55, f"{age} | {position} | {club}", fontsize=11, color="#555555")
    
    # Add stats info (matches, seasons, clubs) with organized layout
    total_matches = player_info.get('total_matches', 0)
    total_seasons = player_info.get('total_seasons', 0)
    total_minutes = player_info.get('total_minutes', 0)
    total_goals = player_info.get('total_goals', 0)
    
    # Format stats with cleaner presentation
    stats_info_text = f"Matches: {total_matches} | Minutes: {total_minutes} | Goals: {total_goals} | Seasons: {total_seasons}"
    ax_info.text(0.02, 0.3, stats_info_text, fontsize=9, color="#666666")
    
    # If player image is available, add it with better positioning and styling
    if player_image_path and os.path.exists(player_image_path):
        try:
            img = mpimg.imread(player_image_path)
            # Add a small inset axes for the image with improved placement
            ax_img = fig.add_axes([0.7, 0.92, 0.22, 0.22], frameon=True)
            ax_img.imshow(img)
            ax_img.axis('off')
            # Add subtle border around image
            ax_img.patch.set_edgecolor('#DDDDDD')
            ax_img.patch.set_linewidth(1)
        except Exception as e:
            logger.error(f"Error loading image: {e}")
    
    # Create the main chart with improved styling
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
    
    # Categorize all stats with improved organization
    categorized_stats = []
    category_dividers = []
    current_pos = 0
    
    # First prioritize Offensive stats (most important for forwards)
    for category_name in ["Offensive", "Progressive", "Defensive", "General"]:
        category_stats = stat_categories.get(category_name, [])
        category_stats_present = [stat for stat in category_stats if stat in all_stats]
        if category_stats_present:
            categorized_stats.extend(category_stats_present)
            current_pos += len(category_stats_present)
            category_dividers.append((current_pos, category_name))
    
    # Handle stats that aren't in any category
    general_stats = [stat for stat in all_stats if stat not in categorized_stats]
    if general_stats:
        categorized_stats = general_stats + categorized_stats
        # Shift all category dividers
        category_dividers = [(pos + len(general_stats), cat) for pos, cat in category_dividers]
        # Add general category if needed
        if general_stats:
            category_dividers.insert(0, (len(general_stats), "Other"))
    
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
    
    # Get actual values if provided with improved formatting
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
    
    # Generate colors for bars based on percentile values with smoother gradient
    bar_colors = []
    for value in percentile_values:
        bar_colors.append(get_percentile_color(value))
    
    # Plot horizontal bars with improved styling
    bars = ax.barh(
        y_positions,
        percentile_values,
        height=0.5,  # More compact bars
        color=bar_colors,
        alpha=0.9,
        edgecolor='white',  # Add subtle white edge
        linewidth=0.3
    )
    
    # Add value labels to bars with better formatting and positioning
    for i, (bar, percentile_val, actual_val) in enumerate(zip(bars, percentile_values, actual_values)):
        if percentile_val > 5:  # Only add text if bar is wide enough
            # Get the current stat name
            stat_name = valid_stats[i]
            
            # The actual_val here is the mean value from the dataframe
            avg_val = round(actual_val, 2)
            
            # Calculate sum by multiplying average by number of matches if info is available
            matches = player_info.get('total_matches', 1)
            if matches > 0:
                sum_val = round(avg_val * matches, 2)
            else:
                sum_val = avg_val  # Fallback if no match data
            
            # Format values to avoid excess precision
            # Remove trailing zeros for cleaner display
            sum_str = f"{sum_val:.1f}".rstrip('0').rstrip('.') if sum_val != int(sum_val) else f"{int(sum_val)}"
            avg_str = f"{avg_val:.1f}".rstrip('0').rstrip('.') if avg_val != int(avg_val) else f"{int(avg_val)}"
            
            # Display the sum and average values
            display_text = f"{sum_str} | {avg_str}"
            
            # Calculate appropriate text position based on bar width
            # Text inside bar for longer bars, outside for shorter ones
            if percentile_val > 25:
                # Inside the bar
                x_pos = percentile_val - 3
                text_color = 'white'
                ha_align = 'right'
            else:
                # Outside the bar
                x_pos = percentile_val + 2
                text_color = '#333333'
                ha_align = 'left'
                
            ax.text(
                x_pos,
                bar.get_y() + bar.get_height()/2,
                display_text,
                va='center',
                ha=ha_align,
                fontsize=7.5,
                fontweight='bold',
                color=text_color
            )
    
    # Set labels and ticks with cleaner styling
    ax.set_yticks(y_positions)
    ax.set_yticklabels(valid_stats, fontsize=8)
    
    # Add subtle label for the x-axis
    ax.set_xlabel('Percentile Rank', fontsize=9, color='#666666')
    
    # Set x-axis range and ticks
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis='x', labelsize=8, colors='#666666')
    ax.tick_params(axis='y', labelsize=8, colors='#333333')
    
    # Add subtle gridlines
    ax.grid(axis='x', linestyle='--', alpha=0.2, color='#888888')
    
    # Add vertical lines at 0, 20, 40, 60, 80, 100 for percentile bands
    for x in [0, 20, 40, 60, 80, 100]:
        ax.axvline(x=x, color='#888888', linestyle='-', alpha=0.15, linewidth=0.8)
    
    # Add category dividers and labels with improved styling
    # Define category colors
    category_colors = {
        "General": '#1ABC9C',  # Teal
        "Other": '#1ABC9C',    # Teal
        "Defensive": '#2E86C1', # Blue
        "Progressive": '#8E44AD', # Purple
        "Offensive": '#D35400'  # Orange
    }
    
    for i, (pos, category) in enumerate(category_dividers):
        if i > 0:  # Skip the first divider
            y_pos = y_positions[pos-1] + 0.5 if pos < len(y_positions) else len(y_positions) - 0.5
            ax.axhline(y=y_pos, color='#888888', linestyle='-', linewidth=0.8, alpha=0.3)
        
        # Calculate middle position for category label
        start_idx = 0 if i == 0 else category_dividers[i-1][0]
        end_idx = pos
        if start_idx < end_idx and end_idx <= len(y_positions):
            mid_pos = (y_positions[start_idx] + y_positions[min(end_idx-1, len(y_positions)-1)]) / 2
            
            # Get category color
            rect_color = category_colors.get(category, '#1ABC9C')
            
            # Add colored rectangle on the right with better positioning
            rect_height = 0.8 * (end_idx - start_idx)
            rect_y = mid_pos - rect_height/2
            
            # Add colored rectangle on the right
            rect = plt.Rectangle((1.01, rect_y), 0.03, rect_height, 
                                transform=ax.transAxes, color=rect_color, alpha=0.8,
                                edgecolor='none', linewidth=0)
            ax.add_patch(rect)
            
            # Add text with improved typography
            ax.text(1.05, mid_pos, category, transform=ax.get_yaxis_transform(), 
                    rotation=270, fontsize=9, fontweight='bold', 
                    ha='center', va='center', color='#333333')
    
    # Add subtle horizontal lines for better readability
    for y in y_positions:
        ax.axhline(y=y-0.3, color='#DDDDDD', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Remove spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_color('#888888')
    ax.spines['left'].set_color('#888888')
    
    # Add percentile color legend at the bottom
    legend_elements = get_percentile_legend_elements()
    
    # Create a separate axes for the legend with better positioning
    legend_ax = fig.add_axes([0.1, 0.02, 0.8, 0.02], frameon=False)
    legend_ax.axis('off')
    legend = legend_ax.legend(
        handles=legend_elements, 
        loc='center', 
        ncol=5, 
        frameon=False, 
        fontsize=7,
        title="", 
        title_fontsize=8
    )
    
    plt.tight_layout()
    # Adjust the main plot to make room for the legend
    plt.subplots_adjust(bottom=0.07)
    
    return fig

# Function to generate radar chart for player comparison
def generate_radar_chart(player_names, player_percentiles, player_colors, player_actual_values=None):
    """
    Generate a radar chart for comparing multiple players based on percentile ranks
    
    Args:
        player_names (list): List of player names
        player_percentiles (list): List of DataFrames containing percentile ranks
        player_colors (list): List of colors for each player
        player_actual_values (list, optional): List of DataFrames containing actual values. Defaults to None.
        
    Returns:
        matplotlib.figure.Figure: The generated radar chart figure
    """
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
        
        # Create figure with polar projection
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': 'polar'})
        
        # Set the starting angle to be at the top (North)
        ax.set_theta_offset(np.pi / 2)
        # Go clockwise
        ax.set_theta_direction(-1)
        
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
        plt.figtext(0.5, 0.965, 'Player Performance Comparison', 
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

# Function to generate a forward player type scatter plot
def generate_forward_type_scatter(player_names, player_percentiles, player_actual_values, player_colors, x_stat, y_stat):
    """
    Generate a scatter plot that categorizes forward players into four types based on their stats
    
    Args:
        player_names (list): List of player names
        player_percentiles (list): List of DataFrames containing percentile ranks
        player_actual_values (list): List of DataFrames containing actual values
        player_colors (list): List of colors for each player
        x_stat (str): Statistic to use for x-axis
        y_stat (str): Statistic to use for y-axis
        
    Returns:
        matplotlib.figure.Figure: The generated scatter plot figure
    """
    try:
        if not player_names or not player_percentiles:
            logger.warning("No player data provided for scatter plot")
            return None
        
        # Create figure with better aspect ratio for visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Set dark background like FM24
        fig.patch.set_facecolor('#333333')
        ax.set_facecolor('#333333')
        
        # Extract the x and y values for each player
        x_values = []
        y_values = []
        labels = []
        colors = []
        hover_texts = []
        
        # Create hover texts for each player
        for i, (name, percentile_df, actual_df) in enumerate(zip(player_names, player_percentiles, player_actual_values)):
            if x_stat in percentile_df.columns and y_stat in percentile_df.columns:
                x_val = float(percentile_df[x_stat].iloc[0]) if not pd.isna(percentile_df[x_stat].iloc[0]) else 0
                y_val = float(percentile_df[y_stat].iloc[0]) if not pd.isna(percentile_df[y_stat].iloc[0]) else 0
                
                # Ensure values stay within plot bounds (with a small buffer)
                x_val = max(5, min(95, x_val))
                y_val = max(5, min(95, y_val))
                
                x_values.append(x_val)
                y_values.append(y_val)
                labels.append(name)
                colors.append(player_colors[i % len(player_colors)])
                
                # Create hover text with player details
                hover_text = f"{name}\n"
                
                # Add x and y stat values
                if x_stat in actual_df.columns:
                    x_actual = float(actual_df[x_stat].iloc[0]) if not pd.isna(actual_df[x_stat].iloc[0]) else 0
                    hover_text += f"{x_stat}: {x_actual:.2f} ({x_val:.0f}%)\n"
                
                if y_stat in actual_df.columns:
                    y_actual = float(actual_df[y_stat].iloc[0]) if not pd.isna(actual_df[y_stat].iloc[0]) else 0
                    hover_text += f"{y_stat}: {y_actual:.2f} ({y_val:.0f}%)\n"
                
                # Add a few more key stats if available
                additional_stats = ["Goals", "Assists", "Shots", "Passes accurate", "Duels won"]
                for stat in additional_stats:
                    if stat != x_stat and stat != y_stat and stat in actual_df.columns:
                        stat_val = float(actual_df[stat].iloc[0]) if not pd.isna(actual_df[stat].iloc[0]) else 0
                        hover_text += f"{stat}: {stat_val:.2f}\n"
                
                hover_texts.append(hover_text)
        
        # Set limits with padding to ensure all points are within the plot area
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        
        # Add thin center lines for quadrant division
        ax.axhline(y=50, color='#666666', linestyle='-', alpha=0.8, linewidth=0.8)
        ax.axvline(x=50, color='#666666', linestyle='-', alpha=0.8, linewidth=0.8)
        
        # Add very subtle grid lines
        ax.grid(True, linestyle='-', alpha=0.1, color='#999999', linewidth=0.5)
        
        # Add quadrant labels in corners with FM24-like style
        label_style = dict(
            fontsize=9,
            ha='left',
            va='top',
            color='#999999',
            fontweight='normal',
        )
        
        # Add quadrant descriptions for forward types with minimalist style
        # ax.text(5, 95, "Deep-Lying Forward", **label_style)
        # ax.text(55, 95, "Advanced Forward", **label_style)
        # ax.text(5, 5, "Poacher", **label_style)
        # ax.text(55, 5, "Pressing Forward", **label_style)
        
        # Add axis labels with clean styling
        ax.set_xlabel(x_stat.upper(), fontsize=9, color='#CCCCCC', labelpad=10)
        ax.set_ylabel(y_stat.upper(), fontsize=9, color='#CCCCCC', labelpad=10)
        
        # Clean up axis ticks - minimize them for minimalism
        ax.tick_params(axis='both', which='major', labelsize=8, colors='#CCCCCC', length=2)
        
        # Customize the tick values
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_yticks([0, 25, 50, 75, 100])
        
        # Plot the scatter points
        sc = ax.scatter(
            x_values, 
            y_values, 
            c='#CCCCCC',  # Light grey dots
            s=80,  # Marker size
            alpha=0.5, 
            edgecolors='none',
            zorder=5,
            picker=5  # Make the points pickable for hover interaction
        )
        
        # Store player info for use in hover event
        player_info_for_hover = dict(zip(range(len(labels)), hover_texts))
        
        # Add player labels with colored background
        annotations = []
        for i, (name, x, y) in enumerate(zip(labels, x_values, y_values)):
            # Ensure label positions are adjusted to stay within plot
            rect_props = dict(
                boxstyle="round,pad=0.3", 
                fc=colors[i],  # Use the player's color
                ec="none",
                alpha=0.8
            )
            
            # Position the annotation to avoid going outside the plot
            y_offset = -15
            if y < 10:  # If close to bottom edge, move label above point
                y_offset = 15
                
            anno = ax.annotate(
                name, 
                (x, y), 
                xytext=(0, y_offset),  # Position label with adjusted offset
                textcoords='offset points', 
                fontsize=9, 
                color='white',
                weight='normal',
                ha='center', 
                va='center',
                bbox=rect_props,
                zorder=10
            )
            annotations.append(anno)
        
        # Add brief descriptions of each forward type - minimalist style
        desc_style = dict(fontsize=8, color='#AAAAAA', ha='center', va='center')
        
        ax.text(25, 75, "Deep-Lying Forward:Creates chances & links play", **desc_style)
        ax.text(75, 75, "Advanced Forward: All-round attacker", **desc_style)
        ax.text(25, 25, "Poacher: Focused on scoring", **desc_style)
        ax.text(75, 25, "Pressing Forward: High work rate & pressing", **desc_style)
        
        # Remove spines for minimalist look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#666666')
        ax.spines['left'].set_color('#666666')
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)
        
        # Create tooltip annotation that will be used for hover text
        tooltip = ax.annotate(
            "", 
            xy=(0, 0), 
            xytext=(20, 20),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", fc="#444444", alpha=0.9, ec="none"),
            color="#FFFFFF",
            fontsize=9,
            ha="left",
            va="bottom",
            visible=False,
            zorder=15
        )
        
        # Define hover behavior
        def hover(event):
            vis = tooltip.get_visible()
            if event.inaxes == ax:
                cont, ind = sc.contains(event)
                if cont:
                    # Get the point index that was hovered over
                    index = ind["ind"][0]
                    # Get player info for the point
                    hover_info = player_info_for_hover.get(index, "")
                    # Update tooltip text and position
                    tooltip.set_text(hover_info)
                    tooltip.xy = (x_values[index], y_values[index])
                    tooltip.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        tooltip.set_visible(False)
                        fig.canvas.draw_idle()
        
        # Connect the hover event to the figure
        fig.canvas.mpl_connect("motion_notify_event", hover)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        logger.error(f"Error generating scatter plot: {str(e)}", exc_info=True)
        # Create a figure with error message
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"Error generating scatter plot: {str(e)}", 
                ha='center', va='center', fontsize=14, color='red')
        ax.axis('off')
        return fig 