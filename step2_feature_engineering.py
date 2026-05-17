"""
French Open Prediction — Step 2: Feature Engineering
=====================================================
Reads  : data/raw_matches.csv  (output of step1_data_collection.py)
Outputs: data/features.csv     (one row per match, ready for model training)
         data/player_snapshot.csv (latest feature values per player, for 2026 prediction)

Run:
    python step2_feature_engineering.py

Expected time: 3-6 minutes (walks through 45,000 matches chronologically)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR  = Path("data")
OUT_DIR   = Path("data")
OUT_DIR.mkdir(exist_ok=True)

CLAY_SURFACE  = "Clay"
ELO_K         = 32        # how fast Elo rating moves after each match
ELO_START     = 1500      # every new player starts here
FORM_WINDOW   = 10        # last N matches for recent form feature
CLAY_WINDOW_YRS = 2       # years of clay history for clay win rate

ROUND_SCORE = {
    "R128": 1, "R64": 2, "R32": 3, "R16": 4,
    "QF": 5, "SF": 6, "F": 7, "W": 8
}

# ── Load clean data ───────────────────────────────────────────────────────────

def load_data():
    path = DATA_DIR / "raw_matches.csv"
    if not path.exists():
        raise FileNotFoundError(
            "data/raw_matches.csv not found.\n"
            "Run step1_data_collection.py first."
        )
    df = pd.read_csv(path, low_memory=False)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.sort_values("match_date").reset_index(drop=True)
    print(f"Loaded {len(df):,} matches from raw_matches.csv")
    return df


# ── Feature 1 & 2: Elo ratings ───────────────────────────────────────────────
#
# We maintain TWO Elo ratings per player:
#   elo_all  → updated after every match on any surface (general skill)
#   elo_clay → updated only after clay matches (clay-specific skill)
#
# We record each player's rating BEFORE the match starts.
# That "before" value becomes the feature — no data leakage.

def expected_score(rating_a, rating_b):
    """Probability that player A beats player B given their Elo ratings."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def compute_elo(df):
    print("Computing Elo ratings...")

    elo_all  = {}   # player_id → current general Elo
    elo_clay = {}   # player_id → current clay-only Elo

    winner_elo_before      = []
    loser_elo_before       = []
    winner_clay_elo_before = []
    loser_clay_elo_before  = []

    for _, row in df.iterrows():
        w       = row["winner_id"]
        l       = row["loser_id"]
        surface = row.get("surface", "")

        # Current ratings (new players start at ELO_START)
        w_gen  = elo_all.get(w, ELO_START)
        l_gen  = elo_all.get(l, ELO_START)
        w_clay = elo_clay.get(w, ELO_START)
        l_clay = elo_clay.get(l, ELO_START)

        # Record BEFORE-match ratings as features
        winner_elo_before.append(w_gen)
        loser_elo_before.append(l_gen)
        winner_clay_elo_before.append(w_clay)
        loser_clay_elo_before.append(l_clay)

        # Update general Elo for both players (all surfaces)
        exp_w          = expected_score(w_gen, l_gen)
        elo_all[w]     = w_gen + ELO_K * (1 - exp_w)
        elo_all[l]     = l_gen + ELO_K * (0 - (1 - exp_w))

        # Update clay Elo only for clay matches
        if surface == CLAY_SURFACE:
            exp_wc         = expected_score(w_clay, l_clay)
            elo_clay[w]    = w_clay + ELO_K * (1 - exp_wc)
            elo_clay[l]    = l_clay + ELO_K * (0 - (1 - exp_wc))

    df["winner_elo"]      = winner_elo_before
    df["loser_elo"]       = loser_elo_before
    df["winner_clay_elo"] = winner_clay_elo_before
    df["loser_clay_elo"]  = loser_clay_elo_before

    print(f"  Done. Example — Nadal clay Elo: "
          f"{elo_clay.get(104745, 'not found'):.0f}")

    return df, elo_all, elo_clay


# ── Feature 3 & 4: Rolling win rates ─────────────────────────────────────────
#
# For each match we compute (looking only at past matches):
#   recent_form    → win rate in last FORM_WINDOW matches (any surface)
#   clay_win_rate  → win rate on clay in last CLAY_WINDOW_YRS years

def compute_rolling_win_rates(df):
    print(f"Computing rolling win rates (last {FORM_WINDOW} matches form, "
          f"last {CLAY_WINDOW_YRS}yr clay rate)...")
    print("  This takes 2-4 minutes — walking through 45,000 matches...")

    # Build per-player match log: player_id → list of (row_idx, date, surface, won)
    player_log = {}
    for idx, row in df.iterrows():
        date    = row["match_date"]
        surface = row.get("surface", "")
        for pid, won in [(row["winner_id"], 1), (row["loser_id"], 0)]:
            if pid not in player_log:
                player_log[pid] = []
            player_log[pid].append((idx, date, surface, won))

    winner_form     = []
    loser_form      = []
    winner_clay_wr  = []
    loser_clay_wr   = []

    for idx, row in df.iterrows():
        date = row["match_date"]

        for pid_col, form_list, clay_list in [
            ("winner_id", winner_form, winner_clay_wr),
            ("loser_id",  loser_form,  loser_clay_wr),
        ]:
            pid     = row[pid_col]
            history = player_log.get(pid, [])

            # All matches BEFORE this one
            past = [(d, s, w) for (i, d, s, w) in history if i < idx]

            # Recent form: last FORM_WINDOW matches, any surface
            recent      = past[-FORM_WINDOW:]
            form_rate   = np.mean([w for _, _, w in recent]) if recent else 0.5

            # Clay win rate: last CLAY_WINDOW_YRS years on clay
            if pd.notna(date):
                cutoff    = date - pd.DateOffset(years=CLAY_WINDOW_YRS)
                clay_hist = [w for d, s, w in past
                             if s == CLAY_SURFACE and pd.notna(d) and d >= cutoff]
            else:
                clay_hist = []
            clay_rate = np.mean(clay_hist) if clay_hist else 0.5

            form_list.append(form_rate)
            clay_list.append(clay_rate)

        # Progress indicator every 5000 rows
        if idx % 5000 == 0 and idx > 0:
            print(f"  {idx:,} / {len(df):,} matches processed...")

    df["winner_form"]       = winner_form
    df["loser_form"]        = loser_form
    df["winner_clay_wr"]    = winner_clay_wr
    df["loser_clay_wr"]     = loser_clay_wr

    print("  Rolling win rates done.")
    return df


# ── Feature 5: Head-to-head (H2H) ────────────────────────────────────────────
#
# For each match, what fraction of past meetings did today's winner win?
# We compute both overall H2H and clay-specific H2H separately.
# Centred at 0.5 so 0 means "no H2H history" (neutral).

def compute_h2h(df):
    print("Computing head-to-head records...")

    meetings = {}   # (min_id, max_id) → list of (date, surface, winner_id)

    h2h_clay_rates    = []
    h2h_overall_rates = []

    for idx, row in df.iterrows():
        w    = row["winner_id"]
        l    = row["loser_id"]
        date = row["match_date"]
        surf = row.get("surface", "")
        pair = (min(w, l), max(w, l))

        past = meetings.get(pair, [])

        # Helper: win rate for player w in a list of past meetings
        def win_rate(history):
            if not history:
                return 0.5
            wins = sum(1 for _, _, winner in history if winner == w)
            return wins / len(history)

        past_clay    = [(d, s, wn) for d, s, wn in past if s == CLAY_SURFACE]
        h2h_clay_rates.append(win_rate(past_clay))
        h2h_overall_rates.append(win_rate(past))

        # Log this match for future lookups
        if pair not in meetings:
            meetings[pair] = []
        meetings[pair].append((date, surf, w))

    df["winner_h2h_clay"]    = h2h_clay_rates
    df["winner_h2h_overall"] = h2h_overall_rates

    print("  H2H done.")
    return df


# ── Feature 6: Roland Garros history ─────────────────────────────────────────
#
# Best round each player has reached at Roland Garros in past editions.
# Encoded as a number: R128=1 ... Final=7 ... Winner=8
# A player with no RG history gets 0.

def compute_rg_history(df):
    print("Computing Roland Garros history...")

    rg_matches = df[df["tourney_name"].str.contains(
        "Roland Garros", case=False, na=False
    )].copy()

    # Build: player_id → list of (date, round_score)
    player_rg = {}
    for _, row in rg_matches.iterrows():
        rnd_score = ROUND_SCORE.get(str(row.get("round", "")), 1)
        date      = row["match_date"]
        for pid_col in ["winner_id", "loser_id"]:
            pid = row[pid_col]
            if pid not in player_rg:
                player_rg[pid] = []
            player_rg[pid].append((date, rnd_score))

    winner_rg = []
    loser_rg  = []

    for idx, row in df.iterrows():
        date = row["match_date"]
        for pid_col, result_list in [
            ("winner_id", winner_rg),
            ("loser_id",  loser_rg),
        ]:
            pid  = row[pid_col]
            hist = [r for d, r in player_rg.get(pid, [])
                    if pd.notna(d) and pd.notna(date) and d < date]
            result_list.append(max(hist) if hist else 0)

    df["winner_rg_best"] = winner_rg
    df["loser_rg_best"]  = loser_rg

    print("  RG history done.")
    return df


# ── Feature 7: Fatigue ────────────────────────────────────────────────────────
#
# Days since each player's last match.
# Fewer days = less rest = more fatigued.
# Capped at 60 (fully rested beyond that).

def compute_fatigue(df):
    print("Computing fatigue (days since last match)...")

    last_played = {}   # player_id → date of most recent match seen so far
    winner_fat  = []
    loser_fat   = []

    for idx, row in df.iterrows():
        date = row["match_date"]
        for pid_col, fat_list in [
            ("winner_id", winner_fat),
            ("loser_id",  loser_fat),
        ]:
            pid = row[pid_col]
            if pid in last_played and pd.notna(date) and pd.notna(last_played[pid]):
                days = (date - last_played[pid]).days
            else:
                days = 30   # default: assume moderately rested if no history
            fat_list.append(min(days, 60))

        # Update last played date for both players
        for pid_col in ["winner_id", "loser_id"]:
            pid = row[pid_col]
            if pd.notna(date):
                last_played[pid] = date

    df["winner_fatigue"] = winner_fat
    df["loser_fatigue"]  = loser_fat

    print("  Fatigue done.")
    return df


# ── Assemble difference features ─────────────────────────────────────────────
#
# The model sees DIFFERENCE features: winner_value - loser_value.
# Positive = winner had the advantage on that feature.
#
# We also create a MIRRORED copy of each row where winner and loser
# are swapped and label = 0. This doubles our training data and teaches
# the model symmetrically — it learns "player A beats player B" AND
# "player B loses to player A" from the same match.

def build_features(df):
    print("Assembling feature dataframe with mirrored rows...")

    rows = []

    for idx, row in df.iterrows():
        base = {
            # Identifiers — not used as model inputs
            "match_idx"     : idx,
            "match_date"    : row["match_date"],
            "tourney_name"  : row.get("tourney_name", ""),
            "surface"       : row.get("surface", ""),
            "round"         : row.get("round", ""),
            "winner_id"     : row["winner_id"],
            "winner_name"   : row.get("winner_name", ""),
            "loser_id"      : row["loser_id"],
            "loser_name"    : row.get("loser_name", ""),

            # ── Difference features (winner minus loser) ──────────────────
            # All positive values = winner had the advantage

            # Elo
            "diff_elo"          : row["winner_elo"]      - row["loser_elo"],
            "diff_clay_elo"     : row["winner_clay_elo"] - row["loser_clay_elo"],

            # Win rates
            "diff_form"         : row["winner_form"]     - row["loser_form"],
            "diff_clay_wr"      : row["winner_clay_wr"]  - row["loser_clay_wr"],

            # H2H (already a rate, centred around 0.5 → shift to 0)
            "diff_h2h_clay"     : row["winner_h2h_clay"]    - 0.5,
            "diff_h2h_overall"  : row["winner_h2h_overall"] - 0.5,

            # Roland Garros history
            "diff_rg_best"      : row["winner_rg_best"]  - row["loser_rg_best"],

            # Fatigue (more days = more rested = advantage)
            "diff_fatigue"      : row["winner_fatigue"]  - row["loser_fatigue"],

            # ATP ranking (lower number = better player, so we flip)
            "diff_rank"         : (row.get("loser_rank",  500) or 500) -
                                  (row.get("winner_rank", 500) or 500),

            # Target: 1 = player A (the actual winner) won this match
            "label"             : 1,
        }
        rows.append(base)

        # Mirrored row — swap winner/loser, flip all diff features, label = 0
        mirror = base.copy()
        for col in ["diff_elo", "diff_clay_elo", "diff_form", "diff_clay_wr",
                    "diff_h2h_clay", "diff_h2h_overall", "diff_rg_best",
                    "diff_fatigue", "diff_rank"]:
            mirror[col] = -mirror[col]
        mirror["winner_id"]   = base["loser_id"]
        mirror["winner_name"] = base["loser_name"]
        mirror["loser_id"]    = base["winner_id"]
        mirror["loser_name"]  = base["winner_name"]
        mirror["label"]       = 0
        rows.append(mirror)

    features = pd.DataFrame(rows)
    print(f"  Total rows (with mirrors): {len(features):,}")
    return features


# ── Player snapshot for 2026 prediction ──────────────────────────────────────
#
# After processing all history, capture each player's CURRENT feature values.
# This snapshot is what we'll feed the model when predicting 2026 matches.

def build_player_snapshot(df, elo_all, elo_clay):
    print("Building player snapshot for 2026 prediction...")

    # Get the most recent row for each player to extract their latest features
    latest_winner = (
        df.sort_values("match_date")
        .groupby("winner_id")
        .last()
        .reset_index()
        [["winner_id", "winner_name", "winner_elo", "winner_clay_elo",
          "winner_form", "winner_clay_wr", "winner_rg_best", "winner_fatigue"]]
        .rename(columns={
            "winner_id"       : "player_id",
            "winner_name"     : "player_name",
            "winner_elo"      : "elo",
            "winner_clay_elo" : "clay_elo",
            "winner_form"     : "form",
            "winner_clay_wr"  : "clay_wr",
            "winner_rg_best"  : "rg_best",
            "winner_fatigue"  : "fatigue",
        })
    )

    latest_loser = (
        df.sort_values("match_date")
        .groupby("loser_id")
        .last()
        .reset_index()
        [["loser_id", "loser_name", "loser_elo", "loser_clay_elo",
          "loser_form", "loser_clay_wr", "loser_rg_best", "loser_fatigue"]]
        .rename(columns={
            "loser_id"       : "player_id",
            "loser_name"     : "player_name",
            "loser_elo"      : "elo",
            "loser_clay_elo" : "clay_elo",
            "loser_form"     : "form",
            "loser_clay_wr"  : "clay_wr",
            "loser_rg_best"  : "rg_best",
            "loser_fatigue"  : "fatigue",
        })
    )

    # Combine and keep most recent row per player
    snapshot = (
        pd.concat([latest_winner, latest_loser], ignore_index=True)
        .sort_values("player_id")
        .drop_duplicates(subset="player_id", keep="last")
    )

    # Override Elo with the final computed values (most accurate)
    snapshot["elo"]      = snapshot["player_id"].map(elo_all).fillna(ELO_START)
    snapshot["clay_elo"] = snapshot["player_id"].map(elo_clay).fillna(ELO_START)

    snapshot = snapshot.sort_values("clay_elo", ascending=False).reset_index(drop=True)

    print(f"  Snapshot covers {len(snapshot):,} players.")
    return snapshot


# ── Verification report ───────────────────────────────────────────────────────

def print_feature_report(features, snapshot):
    clay_features = features[features["surface"] == CLAY_SURFACE]
    feature_cols  = [c for c in features.columns if c.startswith("diff_")]

    print("\n" + "="*55)
    print("  FEATURE ENGINEERING REPORT")
    print("="*55)
    print(f"\n  Total rows (all surfaces, with mirrors): {len(features):,}")
    print(f"  Clay rows for training                 : {len(clay_features):,}")
    print(f"  Feature columns                        : {len(feature_cols)}")

    print(f"\n  Feature ranges (clay matches only):")
    print(f"  {'Feature':<22} {'Min':>8} {'Mean':>8} {'Max':>8}")
    print(f"  {'-'*50}")
    for col in feature_cols:
        s = clay_features[col]
        print(f"  {col:<22} {s.min():>8.2f} {s.mean():>8.2f} {s.max():>8.2f}")

    print(f"\n  Top 15 players by clay Elo (current):")
    print(f"  {'Rank':<6} {'Player':<25} {'Clay Elo':>10} {'Clay WR':>10}")
    print(f"  {'-'*55}")
    for i, row in snapshot.head(15).iterrows():
        print(f"  {i+1:<6} {row['player_name']:<25} "
              f"{row['clay_elo']:>10.0f} {row['clay_wr']:>9.1%}")

    # Label balance check
    label_counts = features["label"].value_counts()
    print(f"\n  Label balance:")
    print(f"    1 (winner won) : {label_counts.get(1, 0):,}")
    print(f"    0 (loser won)  : {label_counts.get(0, 0):,}")
    print("="*55)


# ── Save outputs ──────────────────────────────────────────────────────────────

def save_outputs(features, snapshot):
    # Save clay-only features for model training
    clay_features = features[features["surface"] == CLAY_SURFACE].copy()
    clay_path     = OUT_DIR / "features.csv"
    clay_features.to_csv(clay_path, index=False)
    print(f"\nSaved → {clay_path}  ({clay_path.stat().st_size / 1e6:.1f} MB)")

    # Save full features (all surfaces) — useful for debugging
    all_path = OUT_DIR / "features_all_surfaces.csv"
    features.to_csv(all_path, index=False)
    print(f"Saved → {all_path}")

    # Save player snapshot
    snap_path = OUT_DIR / "player_snapshot.csv"
    snapshot.to_csv(snap_path, index=False)
    print(f"Saved → {snap_path}")

    print("\nNext step: run  step3_model_training.py")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n── Step 2: Feature Engineering ──\n")

    df                  = load_data()
    df, elo_all, elo_clay = compute_elo(df)
    df                  = compute_rolling_win_rates(df)
    df                  = compute_h2h(df)
    df                  = compute_rg_history(df)
    df                  = compute_fatigue(df)

    features  = build_features(df)
    snapshot  = build_player_snapshot(df, elo_all, elo_clay)

    print_feature_report(features, snapshot)
    save_outputs(features, snapshot)


if __name__ == "__main__":
    main()
