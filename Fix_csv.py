"""
Fix_csv.py

Utility to create a cleaned, numeric-only condensed CSV from
`condensed_train.csv` suitable for training: keeps play-level numeric
features, rusher canonical metrics, and up to 5 closest defenders with
relative distance/diffs. Removes textual columns such as names and
college to avoid spurious correlations.

Usage:
	python Fix_csv.py --in condensed_train.csv --out condensed_train_clean.csv

If no arguments are provided, defaults to the file names above.
"""

import argparse
import pandas as pd
import numpy as np
import re
import os


def build_cleaned_csv(in_path: str, out_path: str):
	print(f"Loading '{in_path}'...")
	df = pd.read_csv(in_path, low_memory=False)

	# identifiers and core play features
	identifiers = ['GameId', 'PlayId', 'NflIdRusher']
	core_play_features = ['Quarter', 'Down', 'Distance', 'YardLine', 'DefendersInTheBox']

	# find defender indices present in the DataFrame (Def1_X, Def2_X, ...)
	def_indices = []
	for c in df.columns:
		m = re.match(r"Def(\d+)_X$", c)
		if m:
			def_indices.append(int(m.group(1)))
	def_indices = sorted(def_indices)

	print(f"Found defender slots: {def_indices}")

	cleaned_rows = []

	for _, row in df.iterrows():
		r = {}
		# identifiers
		for idc in identifiers:
			r[idc] = row.get(idc, np.nan)

		# play features
		for col in core_play_features:
			r[col] = row.get(col, np.nan)

		# rusher canonical metrics (prefer canonical 'X'/'Y')
		rush_x = row.get('X', row.get('Rush1_X', np.nan))
		rush_y = row.get('Y', row.get('Rush1_Y', np.nan))
		r['Rusher_X'] = rush_x
		r['Rusher_Y'] = rush_y
		r['Rusher_S'] = row.get('S', row.get('Rush1_S', np.nan))
		r['Rusher_A'] = row.get('A', row.get('Rush1_A', np.nan))
		r['Rusher_Dis'] = row.get('Dis', row.get('Rush1_Dis', np.nan))
		r['Rusher_Dir'] = row.get('Dir', row.get('Rush1_Dir', np.nan))
		# include rusher weight/height (prefer top-level PlayerWeight/Height)
		r['Rusher_Weight'] = row.get('PlayerWeight', row.get('Rush1_PlayerWeight', np.nan))
		r['Rusher_Height'] = row.get('PlayerHeight', row.get('Rush1_PlayerHeight', np.nan))

		# compute distances to defenders and select up to 5 closest
		distances = []
		for idx in def_indices:
			dx = row.get(f'Def{idx}_X', np.nan)
			dy = row.get(f'Def{idx}_Y', np.nan)
			try:
				dist_e = np.sqrt((rush_x - dx) ** 2 + (rush_y - dy) ** 2)
			except Exception:
				dist_e = np.nan
			distances.append((idx, dist_e))

		distances = [d for d in distances if not pd.isna(d[1])]
		distances.sort(key=lambda x: x[1])
		closest = [idx for idx, _ in distances[:5]]

		for slot in range(1, 6):
			if slot <= len(closest):
				didx = closest[slot - 1]
				# keep numeric defender attributes but NOT raw X/Y coordinates
				# include weight/height plus motion metrics
				r[f'Closest_Def{slot}_PlayerWeight'] = row.get(f'Def{didx}_PlayerWeight', np.nan)
				r[f'Closest_Def{slot}_PlayerHeight'] = row.get(f'Def{didx}_PlayerHeight', np.nan)
				r[f'Closest_Def{slot}_S'] = row.get(f'Def{didx}_S', np.nan)
				r[f'Closest_Def{slot}_A'] = row.get(f'Def{didx}_A', np.nan)
				r[f'Closest_Def{slot}_Dis'] = row.get(f'Def{didx}_Dis', np.nan)
				r[f'Closest_Def{slot}_Dir'] = row.get(f'Def{didx}_Dir', np.nan)
				# relative metrics (euclidean distance and diffs)
				dx = row.get(f'Def{didx}_X', np.nan)
				dy = row.get(f'Def{didx}_Y', np.nan)
				ds = row.get(f'Def{didx}_S', np.nan)
				da = row.get(f'Def{didx}_A', np.nan)
				r[f'Rush_ClosestDef{slot}_Dist'] = np.sqrt((rush_x - dx) ** 2 + (rush_y - dy) ** 2) if not pd.isna(rush_x) and not pd.isna(dx) and not pd.isna(rush_y) and not pd.isna(dy) else np.nan
				r[f'Rush_ClosestDef{slot}_S_Diff'] = (r['Rusher_S'] - ds) if not pd.isna(r['Rusher_S']) and not pd.isna(ds) else np.nan
				r[f'Rush_ClosestDef{slot}_A_Diff'] = (r['Rusher_A'] - da) if not pd.isna(r['Rusher_A']) and not pd.isna(da) else np.nan
			else:
				# pad with NaNs for defender numeric columns (no X/Y columns created)
				r[f'Closest_Def{slot}_PlayerWeight'] = np.nan
				r[f'Closest_Def{slot}_PlayerHeight'] = np.nan
				r[f'Closest_Def{slot}_S'] = np.nan
				r[f'Closest_Def{slot}_A'] = np.nan
				r[f'Closest_Def{slot}_Dis'] = np.nan
				r[f'Closest_Def{slot}_Dir'] = np.nan
				r[f'Rush_ClosestDef{slot}_Dist'] = np.nan
				r[f'Rush_ClosestDef{slot}_S_Diff'] = np.nan
				r[f'Rush_ClosestDef{slot}_A_Diff'] = np.nan

		# include target
		r['Yards'] = row.get('Yards', np.nan)
		cleaned_rows.append(r)

	cleaned_df = pd.DataFrame(cleaned_rows)

	# ensure numeric columns are numeric where possible (coerce non-numeric to NaN)
	for col in cleaned_df.columns:
		if col not in identifiers:
			cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')

	print(f"Writing cleaned CSV to '{out_path}' ({len(cleaned_df)} rows)...")
	cleaned_df.to_csv(out_path, index=False)
	print('Done.')


def main():
	parser = argparse.ArgumentParser(description='Create cleaned condensed CSV (numeric-only).')
	parser.add_argument('--in', dest='in_path', default='condensed_train.csv', help='Input condensed CSV path')
	parser.add_argument('--out', dest='out_path', default='condensed_train_clean.csv', help='Output cleaned CSV path')
	args = parser.parse_args()

	if not os.path.exists(args.in_path):
		raise SystemExit(f"Input file not found: {args.in_path}")

	build_cleaned_csv(args.in_path, args.out_path)


if __name__ == '__main__':
	main()

