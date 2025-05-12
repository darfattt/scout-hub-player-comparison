import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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

# Function to get color based on percentile
def get_percentile_color(value):
    """
    Return a color based on the percentile value using a gradient scale.
    
    Args:
        value (float): Percentile value (0-100)
        
    Returns:
        str: Color hex code
    """
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

# Function to get legend elements for percentile colors
def get_percentile_legend_elements():
    """
    Return matplotlib legend elements for percentile colors
    
    Returns:
        list: List of matplotlib Patch objects for legend
    """
    return [
        Patch(facecolor='#CD5C5C', label='0-20'),
        Patch(facecolor='#FF8C00', label='21-40'),
        Patch(facecolor='#FFC107', label='41-60'),
        Patch(facecolor='#9ACD32', label='61-80'),
        Patch(facecolor='#4CAF50', label='81-100')
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
    with open(css_file, 'r') as f:
        return f.read()

# Function to convert image to base64 for HTML embedding
def image_to_base64(image_path):
    """
    Convert an image to base64 encoding for embedding in HTML
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Base64 encoded image
    """
    import base64
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode() 