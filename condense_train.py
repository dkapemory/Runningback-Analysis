"""Create a condensed CSV with one row per play using the rusher's start-of-play row.

This script mirrors the logic used by `Visual.py`: each play is identified by `PlayId` and
the rusher is the row where `NflId == NflIdRusher`. We pick the earliest `TimeSnap`
for the rusher as the representative start-of-play snapshot and write one row per play
to `condensed_train.csv` (or a user-provided path).

Usage:
    python condense_train.py --csv train.csv --out condensed_train.csv

The script streams `train.csv` in chunks to avoid high memory usage.
"""

import argparse
import pandas as pd
import os


def condense(csv_path, out_path, chunksize=500_000, max_plays=None):
    # First read one row to get all column names
    df_sample = pd.read_csv(csv_path, nrows=1)
    all_columns = df_sample.columns.tolist()
    
    # Position-specific columns that we'll replicate for each player
    pos_cols = ['X', 'Y', 'S', 'A', 'Dis', 'Orientation', 'Dir']
    
    # Player-specific columns we'll replicate
    player_cols = ['DisplayName', 'Position', 'PlayerWeight', 'PlayerHeight', 'PlayerBirthDate', 'PlayerCollegeName'] + pos_cols
    reader = pd.read_csv(csv_path, chunksize=chunksize, low_memory=False)
    earliest_frames = {}  # PlayId -> (timestamp, full frame DataFrame)
    chunks = 0
    
    for chunk in reader:
        chunks += 1
        # First find rushers to identify teams
        rusher_rows = chunk[chunk['NflId'] == chunk['NflIdRusher']]
        if rusher_rows.empty:
            print(f"Chunk {chunks}: no rusher rows")
            continue
            
        # Process each play in this chunk
        for pid in rusher_rows['PlayId'].unique():
            play_rows = chunk[chunk['PlayId'] == pid].copy()
            # Parse TimeSnap with explicit ISO format
            play_rows['TimeSnap_dt'] = pd.to_datetime(play_rows['TimeSnap'], 
                                                     format='%Y-%m-%dT%H:%M:%S.%fZ',
                                                     errors='coerce')
            
            # Get earliest timestamp for this play
            ts = play_rows['TimeSnap_dt'].min()
            if pd.isna(ts):
                continue
                
            # Only replace if we don't have this play yet or this is an earlier frame
            if pid not in earliest_frames or ts < earliest_frames[pid][0]:
                earliest_frames[pid] = (ts, play_rows)
        
        print(f"Chunk {chunks}: processed, plays tracked: {len(earliest_frames)}")
        if max_plays is not None and len(earliest_frames) >= max_plays:
            print(f"Reached max_plays={max_plays}; stopping early.")
            break
    
    if len(earliest_frames) == 0:
        raise RuntimeError('No valid plays found. Check CSV and column names.')
    
    # Process each play's earliest frame into a condensed row
    condensed_rows = []
    for pid, (_, frame) in earliest_frames.items():
        # Get rusher's row and team
        rusher_row = frame[frame['NflId'] == frame['NflIdRusher'].iloc[0]].iloc[0]
        rusher_team = rusher_row['Team']
        
        # Split players into rusher's team and opposing team
        rush_team_rows = frame[frame['Team'] == rusher_team]
        def_team_rows = frame[frame['Team'] != rusher_team]
        
        # Start with the rusher's row as base (for play-level info)
        play_dict = rusher_row.to_dict()
        
        # Add position data for each player on rushing team (including rusher)
        for idx, p in enumerate(rush_team_rows.itertuples(), 1):
            prefix = f'Rush{idx}_'
            for col in player_cols:
                play_dict[prefix + col] = getattr(p, col)
        
        # Fill any missing rushing team positions with NaN (up to 11 players)
        for idx in range(len(rush_team_rows) + 1, 12):
            prefix = f'Rush{idx}_'
            for col in player_cols:
                play_dict[prefix + col] = None
        
        # Add position data for each player on defense team
        for idx, p in enumerate(def_team_rows.itertuples(), 1):
            prefix = f'Def{idx}_'
            for col in player_cols:
                play_dict[prefix + col] = getattr(p, col)
        
        # Fill any missing defense positions with NaN (up to 11 players)
        for idx in range(len(def_team_rows) + 1, 12):
            prefix = f'Def{idx}_'
            for col in player_cols:
                play_dict[prefix + col] = None
        
        condensed_rows.append(play_dict)

    # Convert to DataFrame and write CSV
    df = pd.DataFrame(condensed_rows)
    
    # Ensure consistent ordering
    sort_cols = [c for c in ['GameId', 'PlayId'] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)
    
    # Write out the condensed CSV with one row per PlayId
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} condensed play rows to {out_path}")
    
    # Print column preview to help verify structure
    print("\nColumn groups in output:")
    col_groups = {
        'Play Info': ['GameId', 'PlayId', 'Quarter', 'Down', 'Distance', 'YardLine', 'Yards'],
        'Rush Team': [c for c in df.columns if c.startswith('Rush')],
        'Defense Team': [c for c in df.columns if c.startswith('Def')]
    }
    for group, cols in col_groups.items():
        print(f"\n{group}:")
        print(f"  {', '.join(sorted(cols)[:5])}...")
        print(f"  ({len(cols)} columns)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='train.csv', help='Path to train.csv')
    parser.add_argument('--out', type=str, default='condensed_train.csv', help='Output condensed CSV path')
    parser.add_argument('--chunksize', type=int, default=500_000, help='CSV chunksize')
    parser.add_argument('--max-plays', type=int, default=None, help='Stop after this many plays (for testing)')
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"Input CSV not found: {args.csv}")

    condense(args.csv, args.out, chunksize=args.chunksize, max_plays=args.max_plays)


if __name__ == '__main__':
    main()
