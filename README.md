# Football Player Comparison Tool

A Streamlit application for comparing football players based on their performance statistics from Wyscout data.

## Features

- Interactive player selection - compare up to 3 players at a time
- Multiple visualization options:
  - Bar Chart view with percentile ranks
  - Radar Chart view for multi-player comparison
- Categories include: Defensive, Progressive, and Offensive stats
- Color-coded performance indicators (5-level gradient: red - orange - yellow - light green - green)
- Player information cards with age, position, and club
- Export visualizations as high-quality PNG images
- Download player statistics as CSV files

## Requirements

- Python 3.8+
- Streamlit
- Pandas
- Matplotlib
- NumPy
- Pillow

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd <repository-folder>

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. Place your Wyscout CSV files in the `data/wyscout/` directory
   - Files should be named in the format: `Player stats <Player Name>.csv`

2. (Optional) Add player images to the `data/pics/` directory
   - Images should be named to match the player name (e.g., `Player Name.png`)

3. Run the Streamlit application
```bash
streamlit run app.py
```

4. Use the configuration panel to:
   - Select players to compare (up to 3)
   - Choose visualization type (Bar Chart or Radar Chart)
   - Filter metrics by category

5. Download visualizations or data using the download buttons

## Data Structure

The application expects CSV files with the following columns:
- Match
- Competition
- Date
- Position
- Minutes played
- Total actions / successful
- Goals
- Assists
- Shots / on target
- xG
- And various other performance metrics

## Understanding the Visualization

- **Player Cards**: Shows basic player info (name, age, position, club)
- **Percentile Bars**: 
  - Higher percentile (closer to 100) means better performance relative to the other players
  - Color coding:
    - Green (81-100): Excellent performance
    - Light Green (61-80): Good performance
    - Yellow (41-60): Average performance
    - Orange (21-40): Below average performance
    - Red (0-20): Poor performance

- **Radar Chart**:
  - Each axis represents a key performance metric
  - Larger area indicates better overall performance
  - Multiple players shown on the same chart for direct comparison

## Customization

You can customize the stat categories by modifying the `stat_categories` dictionary in the `main()` function.

## License

MIT 