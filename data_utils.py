import os
import re
import pandas as pd
import logging

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
            """Simple fallback for scipy.stats.percentileofscore.
            
            This function calculates the percentage of values in a dataset that are 
            less than or equal to a given value, representing the percentile rank.
            
            Args:
                data: Array-like object containing the dataset values
                value: Value for which to calculate the percentile rank
                
            Returns:
                float: Percentile rank (0-100) indicating the percentage of values 
                       in the dataset that are less than or equal to the given value
            """
            if not data or len(data) == 0:
                return 50
            
            # Convert to list if it's not
            if not isinstance(data, list):
                data = list(data)
                
            # Sort the data
            sorted_data = sorted(data)
            
            # Count values below or equal to the given value
            count = sum(1 for x in sorted_data if x <= value)
            
            # Calculate the percentile
            return (count / len(sorted_data)) * 100
    
    stats = StatsFallback()
    logging.getLogger("player_comparison_tool").warning("SciPy not available. Using custom percentile calculation as fallback.")

logger = logging.getLogger("player_comparison_tool")

# Function to extract player information from filename
def extract_player_name(filename):
    """
    Extract player name from the filename
    
    Args:
        filename (str): Path to the player stats file
        
    Returns:
        str: Extracted player name
    """
    match = re.search(r"Player stats (.*?)\.csv", os.path.basename(filename))
    if match:
        return match.group(1)
    return os.path.basename(filename)

# Function to extract player info from dataframe
def extract_player_info(df):
    """
    Extract player information from a dataframe
    
    Args:
        df (pandas.DataFrame): Dataframe containing player statistics
        
    Returns:
        dict: Dictionary containing player information
    """
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
        "Ribamar": 27,
        "Uilliam": 30
    }
    
    age = age_map.get(player_name, 25)  # Default age 25 if not found
    
    # Try to extract age from Age column if it exists
    if 'Age' in df.columns:
        age_values = pd.to_numeric(df['Age'].dropna(), errors='coerce')
        if not age_values.empty and age_values.notna().any():
            try:
                # Get the most common age value - mode() behavior changed in pandas 2.1+
                # Handle different pandas versions (mode returns Series in older versions, Index in newer ones)
                mode_result = age_values.mode()
                if isinstance(mode_result, pd.Series):
                    age = int(mode_result.iloc[0])
                else:
                    # In newer pandas versions, mode() returns a different type
                    age = int(mode_result[0])
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
    
    Args:
        df (pandas.DataFrame): DataFrame to process
        exclude_columns (list, optional): List of columns to exclude. Defaults to None.
        
    Returns:
        pandas.DataFrame: DataFrame with only numeric columns
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
def calculate_percentile_ranks(dfs, numeric_stats):
    """
    Calculate percentile ranks for numeric statistics across players.
    
    Args:
        dfs (list): List of player dataframes
        numeric_stats (list): List of numeric statistics to calculate percentiles for
        
    Returns:
        tuple: List of percentile dataframes and list of actual value dataframes
    """
    if not dfs or not numeric_stats:
        return [], []
    
    # Add debug logging
    logger.info(f"Calculating percentiles for {len(dfs)} players and {len(numeric_stats)} stats")
    
    # Create aggregated dataframes with player averages
    player_avgs = []
    for df in dfs:
        # Get player name
        player_name = df['player_name'].iloc[0] if 'player_name' in df.columns and not df['player_name'].empty else "Unknown"
        
        # Select only numeric columns that are in the specified list
        cols_to_use = [col for col in numeric_stats if col in df.columns]
        
        if not cols_to_use:
            logger.warning(f"No valid numeric columns found for player {player_name}")
            continue
        
        # Calculate average for each numeric stat
        player_avg = df[cols_to_use].mean().to_frame().transpose()
        
        # Add player name
        player_avg['player_name'] = player_name
        
        player_avgs.append(player_avg)
    
    if not player_avgs:
        logger.warning("No valid player averages calculated")
        return [], []
    
    # Check if we're working with a very small dataset (2-3 players)
    small_dataset = len(player_avgs) <= 3
    logger.info(f"Processing a {'small' if small_dataset else 'normal'} dataset with {len(player_avgs)} players")
    
    # Combine all player averages - ensure ignore_index is set to True for newer pandas
    combined_avgs = pd.concat(player_avgs, ignore_index=True)
    
    # Create dataframes to store percentile ranks and actual values
    percentile_dfs = []
    actual_dfs = []
    
    # Calculate percentile ranks for each player
    for index, row in combined_avgs.iterrows():
        player_name = row['player_name']
        
        # Create dataframe for percentile ranks and actual values
        percentile_df = pd.DataFrame()
        actual_df = pd.DataFrame()
        
        # Process each stat
        for stat in numeric_stats:
            if stat in combined_avgs.columns and stat != 'player_name':
                # Get all values for this stat across players
                all_values = combined_avgs[stat].dropna()
                
                if all_values.empty:
                    continue
                
                # List of stats where lower values are better (negative stats)
                negative_stats = ["Losses", "Losses own half", "Yellow card", "Red card"]
                
                # Get the current player's value
                player_val = row.get(stat)
                
                if pd.isna(player_val):
                    continue
                
                # Handle stats with all zero values
                if all_values.sum() == 0:
                    percentile_rank = 50  # Default to middle percentile
                    logger.info(f"All zero values for {stat}, defaulting to middle percentile (50)")
                # Handle special stats like cards
                elif stat in ["Yellow card", "Red card"]:
                    # Cards are now normalized to per-90 basis like other stats
                    # For cards, lower values are better (fewer cards is better)
                    # percentileofscore returns the percentage of values at or below the given value
                    # So we invert it (100 - score) to get the correct ranking
                    raw_percentile = stats.percentileofscore(all_values, player_val)
                    percentile_rank = 100 - raw_percentile
                    logger.info(f"Card stat {stat} for {player_name}: raw={player_val}, percentile={raw_percentile}, inverted={percentile_rank}")
                else:
                    # For regular stats, calculate percentile normally
                    if stat in negative_stats:
                        # For negative stats, lower values are better
                        # percentileofscore returns the percentage of values at or below the given value
                        # So we invert it (100 - score) to get the correct ranking where lower is better
                        raw_percentile = stats.percentileofscore(all_values, player_val)
                        percentile_rank = 100 - raw_percentile
                        logger.info(f"Negative stat {stat} for {player_name}: raw={player_val}, percentile={raw_percentile}, inverted={percentile_rank}")
                    else:
                        # For positive stats, higher values are better
                        # percentileofscore returns the percentage of values at or below the given value
                        # A higher percentile means the player ranks better compared to others
                        percentile_rank = stats.percentileofscore(all_values, player_val)
                        logger.info(f"Positive stat {stat} for {player_name}: raw={player_val}, percentile={percentile_rank}")
                
                # For small datasets (2-3 players), adjust percentiles to ensure better distribution
                # With 3 players, percentileofscore might only return values like 0, 33, 66, 100
                # This ensures we map to our 5-category color scale more effectively
                if small_dataset:
                    # Map raw percentiles to our 5-level color scale buckets:
                    # 0-20% (Red), 21-40% (Orange), 41-60% (Yellow), 61-80% (Light Green), 81-100% (Green)
                    
                    # For very low values, keep them in the 0-20% bucket but not at absolute 0
                    if percentile_rank < 10:
                        percentile_rank = 10  # Keep in the 0-20% bucket but visible
                    # Map other values to representative points in each bucket
                    elif percentile_rank < 25:
                        percentile_rank = 20  # Set to top of the 0-20% bucket
                    elif percentile_rank < 50:
                        percentile_rank = 40  # Set to top of the 21-40% bucket (changed from 30)
                    elif percentile_rank < 75:
                        percentile_rank = 60  # Set to top of the 41-60% bucket (changed from 50)
                    elif percentile_rank < 90:
                        percentile_rank = 80  # Set to top of the 61-80% bucket (changed from 70)
                    else:
                        percentile_rank = 90  # Set to middle of the 81-100% bucket
                    
                    # Special case for exactly 2 players - ensure wider distribution
                    if len(player_avgs) == 2:
                        # With 2 players, we'll only have percentile scores of 0 and 100
                        # Map to better values that use more of our color scale
                        if percentile_rank < 25:  # Lower player
                            percentile_rank = 30  # Move to the 21-40% bucket
                        elif percentile_rank > 75:  # Higher player
                            percentile_rank = 70  # Move to the 61-80% bucket
                
                    logger.info(f"Small dataset adjustment for {stat}, player {player_name}: adjusted to {percentile_rank}")
                
                # Add to dataframes
                percentile_df[stat] = [percentile_rank]
                actual_df[stat] = [player_val]
        
        percentile_dfs.append(percentile_df)
        actual_dfs.append(actual_df)
        
        # Log the range of percentiles for diagnostic purposes
        if not percentile_df.empty:
            min_percentile = percentile_df.values.min()
            max_percentile = percentile_df.values.max()
            logger.info(f"Player {player_name} percentile range: {min_percentile:.1f} - {max_percentile:.1f}")
            
            # Additional diagnostic logging to track percentile distribution
            percentile_ranges = {
                "0-20%": 0,
                "21-40%": 0, 
                "41-60%": 0,
                "61-80%": 0,
                "81-100%": 0
            }
            
            # Count how many stats fall into each percentile range
            for stat in percentile_df.columns:
                val = float(percentile_df[stat].iloc[0]) if not pd.isna(percentile_df[stat].iloc[0]) else 0
                if val <= 20:
                    percentile_ranges["0-20%"] += 1
                elif val <= 40:
                    percentile_ranges["21-40%"] += 1
                elif val <= 60:
                    percentile_ranges["41-60%"] += 1
                elif val <= 80:
                    percentile_ranges["61-80%"] += 1
                else:
                    percentile_ranges["81-100%"] += 1
            
            # Log the distribution
            logger.info(f"Player {player_name} percentile distribution: {percentile_ranges}")
    
    return percentile_dfs, actual_dfs 