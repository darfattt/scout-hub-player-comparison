# Football Player Comparison Tool

A modern, interactive tool for comparing football players based on their performance statistics from Wyscout data. This application allows coaches, scouts, and analysts to visualize and compare player metrics in a user-friendly interface.

![Screenshot of the application](https://via.placeholder.com/800x450?text=Player+Comparison+Tool)

## Features

- **Multi-player Comparison**: Compare up to 3 players simultaneously
- **Interactive Visualizations**:
  - **Bar Chart View**: Detailed percentile rankings by category
  - **Radar Chart View**: Comparative overview of key metrics
  - **Forward Type Classification**: Scatter plot visualizing player roles
- **Detailed Statistics Table**: Sortable comparison with percentile ranks, sums, and averages
- **Player Categorization**: View metrics grouped by:
  - Offensive (goals, shots, assists, etc.)
  - Progressive (passes, dribbles, crosses, etc.)
  - Defensive (duels, interceptions, recoveries, etc.)
- **Data Insights**:
  - Performance percentile calculation
  - 5-level color-coded rating system
  - Forward player type classification
- **Export Options**:
  - Download high-quality PNG visualizations
  - Export data as CSV for further analysis
- **Responsive Design**: Clean, modern UI for desktop use

## Requirements

- Python 3.8+
- Libraries (automatically installed):
  - Streamlit 1.35.0+
  - Pandas 2.2.0+
  - Matplotlib 3.8.2+
  - NumPy 1.26.3+
  - Pillow 10.1.0+

## Quick Start

### Windows

1. Clone or download this repository
2. Double-click `run.bat` to start the application
3. The application will open in your default web browser

### Manual Installation

```bash
# Clone the repository
git clone <repository-url>
cd football-player-comparison-tool

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## Data Preparation

1. Place your Wyscout CSV files in the `data/wyscout/` directory
   - Files should be named in the format: `Player stats <Player Name>.csv`
   - The data should contain match-by-match statistics

2. (Optional) Add player images to the `data/pics/` directory
   - Images should match player names (e.g., `Player Name.png`)
   - Recommended format: Square PNG images (200x200px)

## Using the Tool

### Configuration

1. Select players to compare (up to 3) from the dropdown menu
2. Choose visualization type:
   - Bar Chart: Detailed view of one player per chart
   - Radar Chart: Overview comparison of all selected players
3. Filter metrics by category (Defensive, Progressive, Offensive)
4. Filter by competitions if needed

### Visualization Options

- **Bar Charts**: Shows percentile ranks for each player with actual values
- **Radar Chart**: Compares all players across key metrics
- **Forward Type Classification**: Maps forwards into four playing styles with FM24-style visualization:
  - Deep-Lying Forward: Creates chances & links play
  - Advanced Forward: All-round attacker  
  - Poacher: Focused on scoring
  - Pressing Forward: High work rate & pressing

### Table View

- Displays detailed metrics for all players
- Toggle between seeing just percentile ranks or detailed stats
- Sortable columns for easier comparison
- Color-coded percentile ranks for visual assessment

## Understanding the Metrics

### Percentile Ranking

The application uses percentile ranking to compare players:

- **0-20% (Red)**: Poor performance
- **21-40% (Orange)**: Below average performance
- **41-60% (Yellow)**: Average performance
- **61-80% (Light Green)**: Good performance
- **81-100% (Green)**: Excellent performance

### Calculation Method

- For small datasets (2-3 players), the tool uses min-max normalization to distribute values across the 0-100 range
- The percentile rank shows how a player compares to others in the current comparison

## Customization

You can customize the stat categories by modifying the `stat_categories` dictionary in the `main()` function of `app.py`.

## Troubleshooting

- **File Encoding Issues**: If you encounter character display problems, ensure your CSV files are encoded as UTF-8 or latin1
- **Image Display Problems**: Check that player image filenames exactly match the player names from the CSV files
- **Performance Issues**: For very large datasets, try filtering by competitions to improve load times

## Contributing

Contributions to improve the tool are welcome:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Data format based on Wyscout statistics
- Visualization techniques inspired by football analytics best practices
- Built with Streamlit framework for Python 