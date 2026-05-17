"""
French Open Prediction — Step 1: Data Collection & Verification
================================================================
Run this AFTER cloning the dataset:
    git clone https://github.com/JeffSackmann/tennis_atp

This script:
  1. Loads all ATP match CSV files (2010–2025)
  2. Cleans and standardises the data
  3. Runs a verification report so you can see exactly what you have
  4. Saves a clean combined file: data/raw_matches.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = Path("tennis_atp")        # folder created by git clone
OUT_DIR  = Path("data")
OUT_DIR.mkdir(exist_ok=True)

YEARS    = range(2010, 2026)         # 2010 → 2025 (2026 data streams in live)

# Columns we actually need — drop everything else to keep the file small
KEEP_COLS = [
    "tourney_id",       # unique tournament ID
    "tourney_name",     # e.g. "Roland Garros"
    "surface",          # Clay / Hard / Grass / Carpet
    "tourney_date",     # YYYYMMDD integer
    "match_num",        # match number within tournament
    "winner_id",        # unique player ID
    "winner_name",      # human-readable name
    "winner_rank",      # ATP ranking at time of match
    "winner_rank_points",
    "loser_id",
    "loser_name",
    "loser_rank",
    "loser_rank_points",
    "round",            # R128, R64, R32, R16, QF, SF, F
    "best_of",          # 3 or 5 sets
    "minutes",          # match duration (useful as fatigue proxy)
    "w_svpt",           # winner serve points
    "w_1stWon",         # winner 1st serve points won
    "w_2ndWon",         # winner 2nd serve points won
    "l_svpt",
    "l_1stWon",
    "l_2ndWon",
    "score",            # e.g. "6-3 6-4"
]


# ── Step 1: Load raw CSV files ────────────────────────────────────────────────

def load_raw_matches():
    """Read yearly CSV files and combine into a single dataframe."""
    frames = []
    missing = []

    print("Loading match files...")
    for yr in YEARS:
        path = DATA_DIR / f"atp_matches_{yr}.csv"
        if not path.exists():
            missing.append(yr)
            continue

        df = pd.read_csv(path, low_memory=False)

        # Keep only columns that exist in this file (some years differ slightly)
        cols = [c for c in KEEP_COLS if c in df.columns]
        df   = df[cols].copy()
        df["source_year"] = yr
        frames.append(df)
        print(f"  {yr}: {len(df):>5,} matches loaded")

    if missing:
        print(f"\n  Warning: missing files for years {missing}")
        print("  Make sure you ran: git clone https://github.com/JeffSackmann/tennis_atp\n")

    if not frames:
        raise FileNotFoundError(
            "No match files found. "
            "Clone the dataset first:\n"
            "  git clone https://github.com/JeffSackmann/tennis_atp"
        )

    data = pd.concat(frames, ignore_index=True)
    print(f"\nTotal: {len(data):,} matches loaded across {len(frames)} years")
    return data


# ── Step 2: Clean and standardise ────────────────────────────────────────────

def clean_data(data):
    """
    Fix data types, fill obvious gaps, add useful derived columns.
    """
    # Parse match date from YYYYMMDD integer → proper datetime
    data["match_date"] = pd.to_datetime(
        data["tourney_date"].astype(str), format="%Y%m%d", errors="coerce"
    )

    # Sort chronologically — critical for feature engineering later
    data = data.sort_values("match_date").reset_index(drop=True)

    # Standardise surface names (occasionally inconsistent)
    surface_map = {
        "clay": "Clay", "hard": "Hard", "grass": "Grass",
        "carpet": "Carpet", "Clay": "Clay", "Hard": "Hard",
        "Grass": "Grass", "Carpet": "Carpet"
    }
    data["surface"] = data["surface"].map(surface_map).fillna(data["surface"])

    # Fill missing rankings with a high number (unranked → 999)
    data["winner_rank"] = pd.to_numeric(data["winner_rank"], errors="coerce").fillna(999)
    data["loser_rank"]  = pd.to_numeric(data["loser_rank"],  errors="coerce").fillna(999)

    # Flag Roland Garros matches specifically
    data["is_roland_garros"] = data["tourney_name"].str.contains(
        "Roland Garros", case=False, na=False
    )

    # Flag clay matches
    data["is_clay"] = data["surface"] == "Clay"

    # Compute serve percentages where data is available
    for prefix in ["w", "l"]:
        svpt  = pd.to_numeric(data.get(f"{prefix}_svpt",  pd.Series()), errors="coerce")
        won1  = pd.to_numeric(data.get(f"{prefix}_1stWon", pd.Series()), errors="coerce")
        won2  = pd.to_numeric(data.get(f"{prefix}_2ndWon", pd.Series()), errors="coerce")
        data[f"{prefix}_serve_pct"] = (won1 + won2) / svpt.replace(0, np.nan)

    print("Data cleaned and standardised.")
    return data


# ── Step 3: Verification report ───────────────────────────────────────────────

def print_verification_report(data):
    """
    Print a human-readable summary so you can confirm the data looks right
    before moving on to feature engineering.
    """
    clay     = data[data["is_clay"]]
    rg       = data[data["is_roland_garros"]]

    print("\n" + "="*55)
    print("  DATA VERIFICATION REPORT")
    print("="*55)

    print(f"\n  Date range   : {data['match_date'].min().date()} → "
          f"{data['match_date'].max().date()}")
    print(f"  Total matches: {len(data):,}")

    print(f"\n  By surface:")
    surface_counts = data["surface"].value_counts()
    for surf, cnt in surface_counts.items():
        pct = cnt / len(data) * 100
        print(f"    {surf:<10} {cnt:>6,}  ({pct:.1f}%)")

    print(f"\n  Clay matches : {len(clay):,}")
    print(f"  Roland Garros: {len(rg):,} matches across all rounds")

    print(f"\n  Roland Garros by year:")
    rg_by_year = rg.groupby("source_year").size()
    for yr, cnt in rg_by_year.items():
        print(f"    {yr}: {cnt} matches")

    print(f"\n  Unique players (overall): {data['winner_id'].nunique():,}")
    print(f"  Unique players (clay)   : {clay['winner_id'].nunique():,}")

    # Top 10 players by clay matches played
    clay_players = pd.concat([
        clay[["winner_id", "winner_name"]].rename(
            columns={"winner_id": "player_id", "winner_name": "player_name"}),
        clay[["loser_id", "loser_name"]].rename(
            columns={"loser_id": "player_id", "loser_name": "player_name"}),
    ])
    top_clay = (
        clay_players.groupby(["player_id", "player_name"])
        .size()
        .reset_index(name="matches")
        .sort_values("matches", ascending=False)
        .head(10)
    )
    print(f"\n  Top 10 players by clay matches played:")
    for _, row in top_clay.iterrows():
        print(f"    {row['player_name']:<25} {row['matches']} matches")

    # Roland Garros champions in the dataset
    rg_finals = rg[rg["round"] == "F"][["source_year", "winner_name", "loser_name"]]
    print(f"\n  Roland Garros finalists in dataset:")
    for _, row in rg_finals.iterrows():
        print(f"    {row['source_year']}: {row['winner_name']} def. {row['loser_name']}")

    # Data quality check
    missing_rank = (data["winner_rank"] == 999).sum() + (data["loser_rank"] == 999).sum()
    missing_date = data["match_date"].isna().sum()
    print(f"\n  Data quality:")
    print(f"    Missing ranks (filled with 999): {missing_rank:,}")
    print(f"    Missing dates                  : {missing_date:,}")
    print(f"    Rows with complete serve stats : "
          f"{data['w_serve_pct'].notna().sum():,}")
    print("="*55)


# ── Step 4: Save clean data ───────────────────────────────────────────────────

def save_clean_data(data):
    out_path = OUT_DIR / "raw_matches.csv"
    data.to_csv(out_path, index=False)
    size_mb = out_path.stat().st_size / 1_000_000
    print(f"\nSaved → {out_path}  ({size_mb:.1f} MB)")
    print("Next step: run  step2_feature_engineering.py")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n── Step 1: Data Collection ──\n")

    data = load_raw_matches()
    data = clean_data(data)
    print_verification_report(data)
    save_clean_data(data)


if __name__ == "__main__":
    main()
