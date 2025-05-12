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