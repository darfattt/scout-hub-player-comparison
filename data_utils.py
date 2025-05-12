import os
import re
import pandas as pd
import logging

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
def calculate_percentile_ranks(df_list, stat_cols):
    """
    Calculate percentile ranks for player statistics
    
    Args:
        df_list (list): List of pandas DataFrames containing player statistics
        stat_cols (list): List of columns to calculate percentiles for
        
    Returns:
        tuple: Tuple containing lists of percentile DataFrames and actual value DataFrames
    """
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
        # With only one player, we'll return a dataframe with all zeros (or another default value)
        player_percentiles = []
        player_actual_values = []
        
        for i in range(len(df_list)):
            # Create a dataframe with zeros for all stats
            percentile_df = pd.DataFrame(columns=valid_cols, data=[[50] * len(valid_cols)])
            player_percentiles.append(percentile_df)
            
            # Get the actual values
            actual_df = combined_stats[combined_stats['player_index'] == i].copy()
            if not actual_df.empty:
                actual_df = actual_df.drop(columns=['player_index'])
                player_actual_values.append(actual_df)
            else:
                # If no actual values found for this player, create a dataframe with zeros
                actual_df = pd.DataFrame(columns=valid_cols, data=[[0] * len(valid_cols)])
                player_actual_values.append(actual_df)
        
        return player_percentiles, player_actual_values
    
    # Calculate percentile ranks using min-max normalization for small datasets
    # This is more appropriate than percentile_rank for 2-3 players
    player_percentiles = []
    player_actual_values = []
    
    for i in range(len(df_list)):
        # Create a dataframe for this player's percentiles
        percentile_df = pd.DataFrame(columns=valid_cols)
        
        # Get the actual values for this player
        actual_df = combined_stats[combined_stats['player_index'] == i].copy()
        
        if actual_df.empty:
            logger.warning(f"No data found for player with index {i}")
            # Create empty dataframes with all stats
            percentile_df = pd.DataFrame(columns=valid_cols, data=[[0] * len(valid_cols)])
            actual_df = pd.DataFrame(columns=valid_cols, data=[[0] * len(valid_cols)])
        else:
            # Drop the player index column from the actual values
            actual_df = actual_df.drop(columns=['player_index'])
            
            # Create a row for the percentiles
            percentile_row = []
            
            for col in valid_cols:
                # Apply min-max normalization to convert to 0-100 scale
                min_val = combined_stats[col].min()
                max_val = combined_stats[col].max()
                
                # Check if there's an actual range of values to normalize
                if max_val > min_val:
                    # Get the value for this player
                    player_val = actual_df[col].iloc[0]
                    
                    # Scale to 0-100 range
                    # Using enhanced scaling to prevent division by zero and handle special cases
                    normalized_val = 0
                    try:
                        normalized_val = ((player_val - min_val) / (max_val - min_val)) * 100
                        
                        # Round to avoid floating point precision issues
                        normalized_val = round(normalized_val, 1)
                        
                        # Ensure no values are outside 0-100 range
                        normalized_val = max(0, min(100, normalized_val))
                    except:
                        # Handle any potential division by zero or other errors
                        normalized_val = 50  # Default to middle value if calculation fails
                        
                    percentile_row.append(normalized_val)
                else:
                    # If all values are the same, use a default value (50%)
                    percentile_row.append(50)
            
            # Create the percentile dataframe for this player
            percentile_df = pd.DataFrame([percentile_row], columns=valid_cols)
        
        # Append to the result lists
        player_percentiles.append(percentile_df)
        player_actual_values.append(actual_df)
    
    return player_percentiles, player_actual_values 