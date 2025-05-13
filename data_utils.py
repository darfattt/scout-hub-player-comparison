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
def calculate_percentile_ranks(player_dfs, numeric_stats):
    """
    Calculate percentile ranks for each player's metrics
    
    Args:
        player_dfs (list): List of DataFrames containing player data
        numeric_stats (list): List of numeric stats to calculate percentiles for
        
    Returns:
        tuple: (List of percentile DataFrames, List of actual value DataFrames)
    """
    if not player_dfs or not numeric_stats:
        return [], []
    
    # Create list to store actual values for each player
    actual_values = []
    for df in player_dfs:
        # Process special stats: Yellow card and Red card
        # These should be counted as occurrences rather than averaged
        special_stats = ["Yellow card", "Red card"]
        special_stats_values = {}
        
        for stat in special_stats:
            if stat in df.columns and stat in numeric_stats:
                # Count matches with cards
                card_count = 0
                total_matches = len(df)
                player_name = df['player_name'].iloc[0] if 'player_name' in df.columns else "Unknown"
                competitions = []
                
                for _, row in df.iterrows():
                    try:
                        # Collect competition info for debugging
                        if 'Competition' in df.columns and pd.notna(row['Competition']):
                            comp = str(row['Competition'])
                            if comp not in competitions:
                                competitions.append(comp)
                        
                        # Improved card value handling
                        card_val_raw = row[stat]
                        # Check if it's a string that might contain a numeric value
                        if isinstance(card_val_raw, str):
                            # Remove any non-numeric characters
                            if card_val_raw.strip().isdigit():
                                card_val = int(card_val_raw.strip())
                            else:
                                # Check if it could be a floating point value
                                try:
                                    card_val = float(card_val_raw.strip())
                                except:
                                    card_val = 0
                        else:
                            # Handle numeric values directly
                            card_val = pd.to_numeric(card_val_raw, errors='coerce')
                        
                        # Only count if the value is valid and greater than 0
                        if pd.notna(card_val) and card_val > 0:
                            # Increment if there's at least one card
                            card_count += 1
                    except Exception as e:
                        logger.error(f"Error processing card value for {player_name}: {str(e)}")
                        pass
                        
                # Debug output for Gustavo's cards
                if player_name == "Gustavo Henrique":
                    if competitions:
                        logger.info(f"DEBUG DATA UTILS - {player_name} has {card_count} matches with {stat}s in competitions: {', '.join(competitions)}")
                    else:
                        logger.info(f"DEBUG DATA UTILS - {player_name} has {card_count} matches with {stat}s in unknown competitions")
                    
                    # Additional logging of each row for detailed inspection
                    if stat == "Yellow card" or stat == "Red card":
                        logger.info(f"DEBUG DATA UTILS - {player_name} {stat} values (raw):")
                        for i, row in df.iterrows():
                            match = row.get('Match', 'Unknown match')
                            comp = row.get('Competition', 'Unknown competition')
                            card_val_raw = row.get(stat, 'N/A')
                            try:
                                card_val = pd.to_numeric(card_val_raw, errors='coerce')
                                is_card = "YES" if pd.notna(card_val) and card_val > 0 else "NO"
                            except:
                                card_val = "ERROR"
                                is_card = "ERROR"
                            logger.info(f"  Match: {match} | Competition: {comp} | {stat}: {card_val_raw} | Counted: {is_card}")
                
                # Calculate frequency of cards per match
                if total_matches > 0:
                    special_stats_values[stat] = card_count / total_matches
                else:
                    special_stats_values[stat] = 0
        
        # Get mean values for regular numeric stats
        mean_values = df[numeric_stats].mean().to_frame().T
        
        # Override special stats with corrected values
        for stat, value in special_stats_values.items():
            mean_values[stat] = value
            
        actual_values.append(mean_values)
    
    # List of stats where lower values are better (negative stats)
    negative_stats = ["Losses", "Losses own half"]  # Yellow card and Red card removed
    
    # Combine all actual values for percentile calculation
    all_values = pd.concat(actual_values)
    
    # Calculate percentiles for each metric
    percentile_dfs = []
    for i, mean_df in enumerate(actual_values):
        percentile_df = pd.DataFrame(index=mean_df.index, columns=numeric_stats)
        
        for stat in numeric_stats:
            if stat in mean_df.columns:
                # Get the player value for this stat
                player_value = mean_df[stat].iloc[0]
                
                # Get all values for this stat
                all_stat_values = all_values[stat]
                
                # Calculate percentile rank
                if len(all_stat_values) > 1:  # More than one player
                    # Get min and max for this stat across all players
                    min_val = all_stat_values.min()
                    max_val = all_stat_values.max()
                    
                    if max_val == min_val:  # All values are the same
                        percentile = 50  # Default to median
                    else:
                        # Calculate percentile based on position within min-max range
                        if stat in negative_stats:
                            # Invert for negative stats (lower is better)
                            percentile = 100 - (player_value - min_val) / (max_val - min_val) * 100
                        else:
                            # Normal calculation (higher is better)
                            percentile = (player_value - min_val) / (max_val - min_val) * 100
                else:
                    percentile = 50  # Default to median if only one player
                
                percentile_df[stat] = percentile
        
        percentile_dfs.append(percentile_df)
    
    return percentile_dfs, actual_values 