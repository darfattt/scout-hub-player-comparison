import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import base64

# Function to get player image
def get_player_image(player_name):
    """
    Get the path to a player's image file. If a player-specific image doesn't exist,
    returns the path to a default image.
    
    Args:
        player_name (str): Name of the player
        
    Returns:
        str or None: Path to the image file, or None if no image is found
    """
    # Check if there's an image file with the player's name in the data/pics directory
    image_path = f"data/pics/{player_name}.png"
    if os.path.exists(image_path):
        return image_path
    
    # If not, check for other image formats
    for ext in ['.png', '.jpeg', '.jpg', '.gif']:
        alt_path = f"data/pics/{player_name}{ext}"
        if os.path.exists(alt_path):
            return alt_path
    
    # If no specific player image is found, use a default image
    default_image = "data/pics/default_player.jpg"
    if os.path.exists(default_image):
        return default_image
    
    # If no default image either, return None
    return None

# Function to get color based on percentile using a more sophisticated gradient
def get_percentile_color(value):
    """
    Return a color based on the percentile value using a refined gradient scale.
    
    Args:
        value (float): Percentile value (0-100)
        
    Returns:
        str: Color hex code
    """
    # Define the color ranges with HEX codes
    if value < 20:
        # Red gradients for poor values (0-20%)
        return '#CD5C5C'  # Indian Red
    elif value < 40:
        # Orange gradients for below average values (21-40%)
        return '#FF8C00'  # Dark Orange
    elif value < 60:
        # Yellow gradients for average values (41-60%)
        return '#FFC107'  # Amber/Yellow
    elif value < 80:
        # Light green gradients for good values (61-80%)
        return '#9ACD32'  # Yellow-Green
    else:
        # Green gradients for excellent values (81-100%)
        return '#4CAF50'  # Green
    
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
    Return matplotlib legend elements for percentile colors with improved styling
    
    Returns:
        list: List of matplotlib Patch objects for legend
    """
    return [
        Patch(facecolor='#CD5C5C', edgecolor='white', linewidth=0.5, label='0-20', alpha=0.9),
        Patch(facecolor='#FF8C00', edgecolor='white', linewidth=0.5, label='21-40', alpha=0.9),
        Patch(facecolor='#FFC107', edgecolor='white', linewidth=0.5, label='41-60', alpha=0.9),
        Patch(facecolor='#9ACD32', edgecolor='white', linewidth=0.5, label='61-80', alpha=0.9),
        Patch(facecolor='#4CAF50', edgecolor='white', linewidth=0.5, label='81-100', alpha=0.9)
    ]

# Function to load CSS
def load_css(css_file):
    """
    Load CSS from a file and return it as a string for Streamlit to use
    
    Args:
        css_file (str): Path to the CSS file
        
    Returns:
        str: CSS content as a string
    """
    try:
        with open(css_file, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error loading CSS file: {str(e)}")
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