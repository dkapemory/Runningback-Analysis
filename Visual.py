import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV (use a chunk if file is huge)
df = pd.read_csv('train.csv')

# Choose a specific frame (example: first frame in file)
frame = df.iloc[0]
game_id = frame['GameId']
play_id = frame['PlayId']
timestamp = frame['TimeHandoff']
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

# Load the CSV (use a chunk if file is huge)
df = pd.read_csv('train.csv')

# Choose a specific play by taking the first row's play (you can change this selection)
frame = df.iloc[0]
game_id = frame['GameId']
play_id = frame['PlayId']

# All rows for this play
play_rows = df[(df['GameId'] == game_id) & (df['PlayId'] == play_id)].copy()

# Convert TimeSnap to datetime for ordering
play_rows['TimeSnap_dt'] = pd.to_datetime(play_rows['TimeSnap'], errors='coerce')

# Attempt to find rusher's frames (most accurate for end of play)
rusher_id = frame.get('NflIdRusher', None)
if pd.notna(rusher_id):
    rusher_rows = play_rows[play_rows['NflId'] == rusher_id].copy()
    if not rusher_rows.empty:
        rusher_rows['TimeSnap_dt'] = pd.to_datetime(rusher_rows['TimeSnap'], errors='coerce')
        start_time = rusher_rows['TimeSnap_dt'].min()
        end_time = rusher_rows['TimeSnap_dt'].max()
    else:
        # fallback to full-play times
        start_time = play_rows['TimeSnap_dt'].min()
        end_time = play_rows['TimeSnap_dt'].max()
else:
    start_time = play_rows['TimeSnap_dt'].min()
    end_time = play_rows['TimeSnap_dt'].max()

start_frame_players = play_rows[play_rows['TimeSnap_dt'] == start_time]
end_frame_players = play_rows[play_rows['TimeSnap_dt'] == end_time]

# Extract team abbreviations and scores from the frame
home_abbr = frame['HomeTeamAbbr']
vis_abbr = frame['VisitorTeamAbbr']
home_score = frame['HomeScoreBeforePlay']
vis_score = frame['VisitorScoreBeforePlay']
yards_gained = frame['Yards']

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
    for _, row in players.iterrows():
        # Resolve player's team abbreviation (row['Team'] is 'home' or 'away')
        player_abbr = team_abbr_by_side.get(row['Team'], row['Team'])
        color = team_color_by_abbr.get(player_abbr, 'gray')
        if row.get('Position') == 'RB':
            ax.scatter(row['X'], row['Y'], marker='D', color='#4fc3f7', s=140, edgecolor='black', zorder=4)
        else:
            ax.scatter(row['X'], row['Y'], marker='o', color=color, s=80, edgecolor='black', zorder=3)
        ax.text(row['X'] + 0.5, row['Y'] + 0.5, row['DisplayName'], fontsize=7, alpha=0.8)

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
    ax.text(left_x, 1.02 + box_height/2, f"{home_abbr}", transform=ax.transAxes, ha='left', va='center', fontsize=10, fontweight='bold', color=team_color_by_abbr.get(home_abbr, 'black'))
    ax.text(left_x + 0.08, 1.02 + box_height/2, f"{home_score}", transform=ax.transAxes, ha='left', va='center', fontsize=10)

    # Visitor badge (right)
    ax.text(right_x - 0.12, 1.02 + box_height/2, f"{vis_score}", transform=ax.transAxes, ha='right', va='center', fontsize=10)
    ax.text(right_x, 1.02 + box_height/2, f"{vis_abbr}", transform=ax.transAxes, ha='right', va='center', fontsize=10, fontweight='bold', color=team_color_by_abbr.get(vis_abbr, 'black'))

    # Yards info in center if requested
    if show_yards:
        ax.text(mid_x, 1.02 + box_height/2, f"Yards: {yards_gained}", transform=ax.transAxes, ha='center', va='center', fontsize=10)

    # Frame title and labels
    time_label = players['TimeSnap'].iloc[0] if len(players) > 0 else 'N/A'
    ax.set_title(f"{title}\nTime: {time_label}")
    ax.set_xlabel('Field X (yards)')
    ax.set_ylabel('Field Y (yards)')
    ax.grid(True, linestyle='--', alpha=0.3)


# Create side-by-side plots: start and end of play
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
plot_frame(axes[0], start_frame_players, 'Start of Play', show_yards=False)
plot_frame(axes[1], end_frame_players, 'End of Play (Tackle)', show_yards=True)

# Legend (use team abbreviations)
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=f'{home_abbr} (Red)', markerfacecolor='red', markersize=10, markeredgecolor='black'),
    Line2D([0], [0], marker='o', color='w', label=f'{vis_abbr} (Green)', markerfacecolor='green', markersize=10, markeredgecolor='black'),
    Line2D([0], [0], marker='D', color='w', label='Runningback', markerfacecolor='#4fc3f7', markersize=12, markeredgecolor='black'),
    mpatches.Patch(color='lightblue', alpha=0.3, label='End Zone'),
    mpatches.Patch(color='#d0f5d8', alpha=1, label='Field (Grass)')
]
fig.legend(handles=legend_elements, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 0.98))
plt.subplots_adjust(top=0.88)
plt.tight_layout()
plt.show()