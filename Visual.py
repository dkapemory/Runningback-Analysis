import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

# Load the CSV with low_memory=False to handle mixed types
df = pd.read_csv('train.csv', low_memory=False)

def get_unique_games():
    """Get a list of all unique games in the dataset"""
    unique_games = df[['GameId', 'HomeTeamAbbr', 'VisitorTeamAbbr']].drop_duplicates()
    return unique_games.sort_values('GameId')

def select_game(game_number=None):
    """Select a game by its index number (1-based) in the sorted unique games list"""
    unique_games = get_unique_games()
    total_games = len(unique_games)
    
    if game_number is None:
        # List all available games
        print("\nAvailable games:")
        for idx, (_, row) in enumerate(unique_games.iterrows(), 1):
            print(f"{idx}. Game ID: {row['GameId']} - {row['VisitorTeamAbbr']} @ {row['HomeTeamAbbr']}")
        game_number = int(input(f"\nEnter game number (1-{total_games}): "))
    
    # Validate input
    if not 1 <= game_number <= total_games:
        raise ValueError(f"Game number must be between 1 and {total_games}")
    
    # Get the selected game
    selected_game = unique_games.iloc[game_number - 1]
    game_id = selected_game['GameId']
    
    # Get the first play of this game
    frame = df[df['GameId'] == game_id].iloc[0]
    return frame

# Choose a specific game by taking user input
frame = select_game()
game_id = frame['GameId']

# Get all running plays for this game
game_plays = df[df['GameId'] == game_id].copy()
unique_plays = game_plays[['PlayId', 'NflIdRusher', 'Yards']].drop_duplicates()
total_plays = len(unique_plays)
print(f"\nFound {total_plays} plays in this game.")

# Extract initial frame information
game_play_data = df[df['GameId'] == game_id].copy()
initial_play_info = game_play_data.iloc[0]

# Extract team abbreviations and scores from the initial frame
home_abbr = initial_play_info['HomeTeamAbbr']
vis_abbr = initial_play_info['VisitorTeamAbbr']
home_score = initial_play_info['HomeScoreBeforePlay']
vis_score = initial_play_info['VisitorScoreBeforePlay']

# Map 'home'/'away' to actual abbreviations
team_abbr_by_side = {'home': home_abbr, 'away': vis_abbr}

# Color mapping by abbreviation (keeps legend/colors consistent)
team_color_by_abbr = {home_abbr: 'red', vis_abbr: 'green'}

def plot_frame(ax, players, title, show_yards=False):
    # Set axes and field
    ax.set_xlim(-10, 110)
    ax.set_ylim(0, 53.3)
    ax.set_facecolor('#d0f5d8')  # light green
    # end zones
    ax.axvspan(-10, 0, color='lightblue', alpha=0.3)
    ax.axvspan(100, 110, color='lightblue', alpha=0.3)

    # Plot players
    # Get the actual rusher's ID for this play
    rusher_id = players['NflIdRusher'].iloc[0] if len(players) > 0 else None
    
    for _, row in players.iterrows():
        # Resolve player's team abbreviation (row['Team'] is 'home' or 'away')
        player_abbr = team_abbr_by_side.get(row['Team'], row['Team'])
        color = team_color_by_abbr.get(player_abbr, 'gray')
        
        # Only mark as RB if this player is the actual rusher
        is_rusher = row['NflId'] == rusher_id
        if is_rusher:
            ax.scatter(row['X'], row['Y'], marker='D', color='#4fc3f7', s=140, edgecolor='black', zorder=4)
            # Print rusher info for debugging
            print(f"\nRusher Info: {row['DisplayName']} (ID: {row['NflId']}, Position: {row['Position']})")
        else:
            ax.scatter(row['X'], row['Y'], marker='o', color=color, s=80, edgecolor='black', zorder=3)
            # If we see another RB, print their info
            if row.get('Position') == 'RB':
                print(f"Other RB found: {row['DisplayName']} (ID: {row['NflId']}, Position: {row['Position']})")
        
        ax.text(row['X'] + 0.5, row['Y'] + 0.5, f"{row['DisplayName']} ({row['Position']})", fontsize=7, alpha=0.8)

    # Scoreboard / title
    # Draw a rounded scoreboard box at top-center of the axes
    box_width = 0.5
    box_height = 0.12
    bbox = FancyBboxPatch((0.5 - box_width/2, 1.02), box_width, box_height,
                         transform=ax.transAxes, boxstyle="round,pad=0.02", fc='white', ec='black', linewidth=1, alpha=0.95)
    ax.add_patch(bbox)

    # Team badges and scores inside the box
    left_x = 0.5 - box_width/2 + 0.02
    mid_x = 0.5
    right_x = 0.5 + box_width/2 - 0.02

    # Home badge (left)
    ax.text(left_x, 1.02 + 2*box_height/3, f"{home_abbr}", transform=ax.transAxes, ha='left', va='center', fontsize=10, fontweight='bold', color=team_color_by_abbr.get(home_abbr, 'black'))
    ax.text(left_x + 0.08, 1.02 + 2*box_height/3, f"{home_score}", transform=ax.transAxes, ha='left', va='center', fontsize=10)

    # Visitor badge (right)
    ax.text(right_x - 0.12, 1.02 + 2*box_height/3, f"{vis_score}", transform=ax.transAxes, ha='right', va='center', fontsize=10)
    ax.text(right_x, 1.02 + 2*box_height/3, f"{vis_abbr}", transform=ax.transAxes, ha='right', va='center', fontsize=10, fontweight='bold', color=team_color_by_abbr.get(vis_abbr, 'black'))

    # Yards info in center if requested
    if show_yards:
        ax.text(mid_x, 1.02 + box_height*0.7, f"Yards Gained: {yards_gained}", transform=ax.transAxes, ha='center', va='center', 
                fontsize=10, fontweight='bold', bbox=dict(facecolor='yellow', alpha=0.3, edgecolor=None, pad=3))

    # Frame title and labels
    time_str = players['TimeSnap'].iloc[0] if len(players) > 0 else 'N/A'
    # Convert UTC timestamp to more readable format
    try:
        time_obj = pd.to_datetime(time_str)
        game_clock = players['GameClock'].iloc[0] if len(players) > 0 else 'N/A'
        quarter = players['Quarter'].iloc[0] if len(players) > 0 else 'N/A'
        time_label = f"Quarter: {quarter}  |  Game Clock: {game_clock}"
    except:
        time_label = "Time not available"
    
    ax.set_title(f"{title}\n{time_label}")
    ax.set_xlabel('Field X (yards)')
    ax.set_ylabel('Field Y (yards)')
    ax.grid(True, linestyle='--', alpha=0.3)


# Let the user choose which play to view
print("\nAvailable plays:")
play_summaries = []
for idx, (_, play_info) in enumerate(unique_plays.iterrows(), 1):
    # try to find rusher name
    rusher_id = play_info['NflIdRusher']
    play_id = play_info['PlayId']
    rrows = game_plays[(game_plays['PlayId'] == play_id) & (game_plays['NflId'] == rusher_id)]
    rusher_name = rrows['DisplayName'].iloc[0] if not rrows.empty else 'Unknown'
    yards = play_info['Yards']
    play_summaries.append((play_id, rusher_name, yards))
    print(f"{idx}. PlayId: {play_id} - Rusher: {rusher_name} - Yards: {yards}")

selected = None
while selected is None:
    try:
        choice = int(input(f"\nEnter play number to view (1-{total_plays}): "))
        if 1 <= choice <= total_plays:
            selected = choice
        else:
            print(f"Please enter a number between 1 and {total_plays}.")
    except ValueError:
        print("Invalid input. Please enter a number.")

# Show the selected play
sel_play = play_summaries[selected - 1]
play_id = sel_play[0]
rusher_name = sel_play[1]
yards_gained = sel_play[2]

play_rows = game_plays[game_plays['PlayId'] == play_id].copy()
play_rows['TimeSnap_dt'] = pd.to_datetime(play_rows['TimeSnap'], errors='coerce')
start_time = play_rows['TimeSnap_dt'].min()
end_time = play_rows['TimeSnap_dt'].max()
end_frame_players = play_rows[play_rows['TimeSnap_dt'] == end_time]

# Create plot for the selected play
fig, ax = plt.subplots(1, 1, figsize=(12, 7))
play_title = f'Play {selected}/{total_plays} - Start of Play'
plot_frame(ax, end_frame_players, play_title, show_yards=True)

# Legend (use team abbreviations)
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=f'{home_abbr} (Red)', markerfacecolor='red', markersize=10, markeredgecolor='black'),
    Line2D([0], [0], marker='o', color='w', label=f'{vis_abbr} (Green)', markerfacecolor='green', markersize=10, markeredgecolor='black'),
    Line2D([0], [0], marker='D', color='w', label=f'Rusher: {rusher_name}', markerfacecolor='#4fc3f7', markersize=12, markeredgecolor='black'),
    mpatches.Patch(color='lightblue', alpha=0.3, label='End Zone'),
    mpatches.Patch(color='#d0f5d8', alpha=1, label='Field (Grass)')
]
ax.legend(handles=legend_elements, loc='upper right')
plt.tight_layout()
plt.show()