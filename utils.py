import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import base64
import logging
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

logger = logging.getLogger("player_comparison_tool")

# Function to get player image
def get_player_image(player_name):
    """
    Check if a player image exists and return the path if found.
    
    Args:
        player_name (str): Name of the player
        
    Returns:
        str: Path to player image if found, None otherwise
    """
    # Check various image formats and paths
    image_extensions = ['jpg', 'png', 'jpeg']
    
    # Base path for player images
    base_dir = os.path.join('data', 'player_images')
    os.makedirs(base_dir, exist_ok=True)
    
    # Check for images with player name
    for ext in image_extensions:
        # Try the direct name
        image_path = os.path.join(base_dir, f"{player_name}.{ext}")
        if os.path.exists(image_path):
            return image_path
        
        # Try lowercase
        image_path = os.path.join(base_dir, f"{player_name.lower()}.{ext}")
        if os.path.exists(image_path):
            return image_path
        
        # Try without spaces
        image_path = os.path.join(base_dir, f"{player_name.replace(' ', '')}.{ext}")
        if os.path.exists(image_path):
            return image_path
        
        # Try lowercase without spaces
        image_path = os.path.join(base_dir, f"{player_name.lower().replace(' ', '')}.{ext}")
        if os.path.exists(image_path):
            return image_path
    
    # If no image found, return None
    return None

# Function to get color based on percentile using a more sophisticated gradient
def get_percentile_color(percentile_rank, stat_name=None):
    """
    Get color based on percentile rank using a smoother color gradient.
    
    Args:
        percentile_rank (float): Percentile rank (0-100)
        stat_name (str, optional): Name of the stat, to handle inverted colors for negative stats
        
    Returns:
        str: Hex color code
    """
    # List of negative stats where lower values are better
    negative_stats = ["Losses", "Losses own half"]  # Yellow card and Red card removed
    
    # For negative stats, invert the percentile for color coding
    if stat_name in negative_stats:
        percentile_rank = 100 - percentile_rank
    
    # Ensure percentile is at least 1 for color coding
    percentile_rank = max(percentile_rank, 1)
    
    if percentile_rank >= 90:
        return '#1a9641'  # Dark green
    elif percentile_rank >= 70:
        return '#73c378'  # Medium green
    elif percentile_rank >= 50:
        return '#f9d057'  # Better yellow (more readable than previous '#ffffbf')
    elif percentile_rank >= 30:
        return '#fc8d59'  # Light orange
    else:
        return '#d73027'  # Red

# Function for smoother color transitions (unused but available for future enhancement)
def get_smooth_percentile_color(value):
    """
    Return a color based on the percentile value using a continuous gradient scale.
    
    Args:
        value (float): Percentile value (0-100)
        
    Returns:
        tuple: RGB color tuple
    """
    # Define color stops (in RGB)
    colors = [
        (205, 92, 92),   # Red (0%)
        (255, 140, 0),   # Dark Orange (25%)
        (255, 193, 7),   # Yellow (50%)
        (154, 205, 50),  # Light Green (75%)
        (76, 175, 80)    # Green (100%)
    ]
    
    # Normalize value to 0-1 range
    normalized = value / 100.0
    
    # Get position between color stops
    idx = min(int(normalized * 4), 3)
    fraction = (normalized * 4) - idx
    
    # Interpolate between two closest colors
    r = colors[idx][0] + fraction * (colors[idx+1][0] - colors[idx][0])
    g = colors[idx][1] + fraction * (colors[idx+1][1] - colors[idx][1])
    b = colors[idx][2] + fraction * (colors[idx+1][2] - colors[idx][2])
    
    # Convert to hex color code
    return f'#{int(r):02x}{int(g):02x}{int(b):02x}'

# Function to get legend elements for percentile colors
def get_percentile_legend_elements():
    """
    Create legend elements for percentile colors
    
    Returns:
        list: List of Patch objects for the legend
    """
    legend_elements = [
        mpatches.Patch(facecolor='#d73027', edgecolor='none', label='0-20%'),
        mpatches.Patch(facecolor='#fc8d59', edgecolor='none', label='20-40%'),
        mpatches.Patch(facecolor='#f9d057', edgecolor='none', label='40-60%'),
        mpatches.Patch(facecolor='#73c378', edgecolor='none', label='60-80%'),
        mpatches.Patch(facecolor='#1a9641', edgecolor='none', label='80-100%')
    ]
    return legend_elements

# Function to load CSS
def load_css(file_name):
    """
    Load CSS from the specified file.
    
    Args:
        file_name (str): Name of the CSS file in the styles directory
        
    Returns:
        str: CSS content
    """
    try:
        # Try to read the specified file
        with open(file_name, 'r') as f:
            css = f.read()
        return css
    except Exception as e:
        # If the file doesn't exist or can't be read, return empty string
        logger.error(f"Error loading CSS file '{file_name}': {str(e)}", exc_info=True)
        return ""

# Function to convert image to base64 for HTML embedding
def image_to_base64(image_path):
    """
    Convert an image to base64 encoding for embedding in HTML
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Base64 encoded image
    """
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        print(f"Error encoding image: {str(e)}")
        return ""

# Function to format numeric values for better display 
def format_stat_value(value, precision=1):
    """
    Format a numeric value with appropriate precision, removing trailing zeros
    
    Args:
        value (float): The numeric value to format
        precision (int): Number of decimal places to display
        
    Returns:
        str: Formatted value as string
    """
    if value is None or np.isnan(value):
        return "0"
        
    # If the value is an integer, format without decimal places
    if float(value).is_integer():
        return f"{int(value)}"
    
    # Format with specified precision and remove trailing zeros
    formatted = f"{value:.{precision}f}".rstrip('0').rstrip('.')
    
    return formatted 