"""
French Open Prediction — Step 3: Model Training (128-player draw, fast simulation)
===================================================================================
Reads  : data/features.csv         (output of step2_feature_engineering.py)
         data/player_snapshot.csv  (current player stats)
Outputs: models/xgboost_model.pkl  (trained model)
         models/feature_importance.png
         models/predictions_2026.png
         models/model_report.txt
         data/predictions_2026.csv

Run:
    python step3_model_training.py

Speed fix:
    All 128x127 matchup probabilities are pre-computed in ONE batch model call
    before the simulation starts. The simulation then just does instant
    dictionary lookups instead of calling the model 640,000 times.
    Runtime: ~30 seconds instead of 2+ hours.
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, log_loss, brier_score_loss, roc_auc_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR  = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "diff_elo",
    "diff_clay_elo",
    "diff_form",
    "diff_clay_wr",
    "diff_h2h_clay",
    "diff_h2h_overall",
    "diff_rg_best",
    "diff_fatigue",
    "diff_rank",
]

TARGET_COL = "label"

# Players NOT playing in 2026 (retired or withdrawn)
EXCLUDED_PLAYERS = [
    "Carlos Alcaraz",         # wrist injury — withdrawn
    "Rafael Nadal",           # retired
    "Roger Federer",          # retired
    "Andy Murray",            # retired
    "Juan Martin del Potro",  # retired
    "Robin Soderling",        # retired
    "Stan Wawrinka",          # retired
    "Marin Cilic",            # retired
    "Kei Nishikori",          # retired
    "Gael Monfils",           # retired
    "Tomas Berdych",          # retired
    "Milos Raonic",           # retired
    "Nick Kyrgios",           # retired
]

DRAW_SIZE     = 128    # real French Open main draw size
DISPLAY_TOP   = 20     # show only top N in results table
N_SIMULATIONS = 10000  # Monte Carlo simulation count


# ── Load data ─────────────────────────────────────────────────────────────────

def load_features():
    path = DATA_DIR / "features.csv"
    if not path.exists():
        raise FileNotFoundError(
            "data/features.csv not found.\n"
            "Run step2_feature_engineering.py first."
        )
    df = pd.read_csv(path, low_memory=False)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    print(f"Loaded {len(df):,} clay match rows (features.csv)")
    return df


def load_snapshot():
    path = DATA_DIR / "player_snapshot.csv"
    if not path.exists():
        raise FileNotFoundError(
            "data/player_snapshot.csv not found.\n"
            "Run step2_feature_engineering.py first."
        )
    df = pd.read_csv(path)
    df = df[~df["player_name"].isin(EXCLUDED_PLAYERS)].reset_index(drop=True)
    print(f"Loaded {len(df):,} active players (player_snapshot.csv)")
    return df


# ── Train / test split by date ────────────────────────────────────────────────
#
# Always split by DATE not randomly.
# Train on past (2010-2023), test on recent (2024-2025).
# This prevents data leakage and mirrors real-world usage.

def split_data(df):
    train_df = df[df["match_date"] <  "2024-01-01"].copy()
    test_df  = df[df["match_date"] >= "2024-01-01"].copy()

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_test  = test_df[FEATURE_COLS]
    y_test  = test_df[TARGET_COL]

    print(f"\nTrain set : {len(X_train):,} rows  (2010-2023)")
    print(f"Test set  : {len(X_test):,}  rows  (2024-2025)")
    return X_train, y_train, X_test, y_test


# ── Baseline ──────────────────────────────────────────────────────────────────
#
# Simplest possible prediction: always pick the higher-ranked player.
# Our model must beat this to prove it is useful.

def evaluate_baseline(X_test, y_test):
    print("\n── Baseline: always pick higher-ranked player ──")
    baseline_preds = (X_test["diff_rank"] > 0).astype(int)
    acc = accuracy_score(y_test, baseline_preds)
    print(f"  Baseline accuracy: {acc:.1%}")
    return acc


# ── Train XGBoost ─────────────────────────────────────────────────────────────

def train_xgboost(X_train, y_train):
    print("\n── Training XGBoost model ──")

    model = xgb.XGBClassifier(
        n_estimators     = 500,
        max_depth        = 4,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 5,
        gamma            = 0.1,
        reg_alpha        = 0.1,
        reg_lambda       = 1.0,
        eval_metric      = "logloss",
        random_state     = 42,
        n_jobs           = -1,
    )

    print("  Running 5-fold cross-validation...")
    cv        = StratifiedKFold(n_splits=5, shuffle=False)
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=cv, scoring="roc_auc", n_jobs=-1
    )
    print(f"  CV ROC-AUC : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    model.fit(X_train, y_train, verbose=False)
    print("  XGBoost training complete.")
    return model


# ── Train Logistic Regression ─────────────────────────────────────────────────

def train_logistic(X_train, y_train):
    print("\n── Training Logistic Regression (comparison) ──")
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    lr       = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(X_scaled, y_train)
    print("  Logistic Regression training complete.")
    return lr, scaler


# ── Evaluate models ───────────────────────────────────────────────────────────

def evaluate_models(xgb_model, lr_model, lr_scaler,
                    X_train, y_train, X_test, y_test, baseline_acc):
    print("\n── Model Evaluation on Test Set (2024-2025) ──")

    results = {}

    for name, model, X_te in [
        ("XGBoost",             xgb_model, X_test),
        ("Logistic Regression", lr_model,  lr_scaler.transform(X_test)),
    ]:
        preds = model.predict(X_te)
        probs = model.predict_proba(X_te)[:, 1]

        acc   = accuracy_score(y_test, preds)
        auc   = roc_auc_score(y_test, probs)
        ll    = log_loss(y_test, probs)
        brier = brier_score_loss(y_test, probs)

        results[name] = {
            "accuracy"  : acc,
            "roc_auc"   : auc,
            "log_loss"  : ll,
            "brier"     : brier,
        }

        print(f"\n  {name}:")
        print(f"    Accuracy    : {acc:.1%}   "
              f"(baseline: {baseline_acc:.1%}  |  "
              f"improvement: {(acc - baseline_acc):+.1%})")
        print(f"    ROC-AUC     : {auc:.4f}  "
              f"(random=0.50, perfect=1.00)")
        print(f"    Log Loss    : {ll:.4f}   (lower is better)")
        print(f"    Brier Score : {brier:.4f}  (lower is better)")

    return results


# ── Feature importance chart ──────────────────────────────────────────────────

def plot_feature_importance(model, output_path):
    importance = pd.DataFrame({
        "feature"    : FEATURE_COLS,
        "importance" : model.feature_importances_,
    }).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))

    # Top 3 features in teal, rest in grey
    colors = [
        "#1D9E75" if i >= len(FEATURE_COLS) - 3 else "#B4B2A9"
        for i in range(len(importance))
    ]

    ax.barh(importance["feature"], importance["importance"], color=colors)
    ax.set_xlabel("Feature importance (gain)", fontsize=11)
    ax.set_title(
        "What drives French Open 2026 match predictions?",
        fontsize=12, fontweight="bold"
    )
    ax.spines[["top", "right"]].set_visible(False)

    for i, (val, _) in enumerate(
        zip(importance["importance"], importance["feature"])
    ):
        ax.text(val + 0.001, i, f"{val:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {output_path}")


# ── Pre-compute matchup table (THE KEY SPEED FIX) ────────────────────────────
#
# Instead of calling model.predict_proba() inside the simulation loop
# (which would be 640,000 individual calls = 2+ hours), we:
#   1. Build ALL 128x127 = 16,256 matchup feature rows at once
#   2. Call model.predict_proba() ONCE on all rows as a batch
#   3. Store results in a dict: (p1, p2) -> probability
#
# The simulation then just does instant dictionary lookups.
# Total model calls: 16,256 (once) instead of 640,000 (repeated)

def build_matchup_table(players, snapshot, model):
    """
    Pre-compute win probability for every possible pair of players.
    Returns dict: (player1, player2) -> probability player1 wins.
    """
    print("  Pre-computing all matchup probabilities...")

    rows  = []   # feature rows, one per matchup
    pairs = []   # (p1, p2) pairs in same order as rows

    for p1 in players:
        for p2 in players:
            if p1 == p2:
                continue

            s1 = snapshot[snapshot["player_name"] == p1]
            s2 = snapshot[snapshot["player_name"] == p2]

            if s1.empty or s2.empty:
                # Unknown player — neutral features
                rows.append([0.0] * len(FEATURE_COLS))
            else:
                s1 = s1.iloc[0]
                s2 = s2.iloc[0]
                rows.append([
                    s1["elo"]      - s2["elo"],           # diff_elo
                    s1["clay_elo"] - s2["clay_elo"],       # diff_clay_elo
                    s1["form"]     - s2["form"],           # diff_form
                    s1["clay_wr"]  - s2["clay_wr"],        # diff_clay_wr
                    0.0,                                   # diff_h2h_clay
                    0.0,                                   # diff_h2h_overall
                    s1.get("rg_best", 0) - s2.get("rg_best", 0),  # diff_rg_best
                    0.0,                                   # diff_fatigue
                    (s2.get("rank", 200) or 200) -
                    (s1.get("rank", 200) or 200),          # diff_rank
                ])

            pairs.append((p1, p2))

    # ONE batch prediction call for all matchups
    X     = pd.DataFrame(rows, columns=FEATURE_COLS)
    probs = model.predict_proba(X)[:, 1]

    # Build lookup dict
    table = {pair: float(prob) for pair, prob in zip(pairs, probs)}

    print(f"  Done. {len(table):,} matchups pre-computed in one batch call.")
    return table


# ── Monte Carlo simulation (fast — uses lookup table) ────────────────────────
#
# Simulates the full 128-player bracket N times.
# Each match: look up pre-computed probability, randomly pick winner.
# No model calls inside the loop — just dict lookups (microseconds each).

def monte_carlo_simulation(players, snapshot, model):
    print(f"\n── Monte Carlo Simulation ──")
    print(f"  Players in draw : {len(players)}")
    print(f"  Simulations     : {N_SIMULATIONS:,}")

    # Pre-compute ALL matchups once before simulation starts
    table = build_matchup_table(players, snapshot, model)

    print(f"  Running {N_SIMULATIONS:,} simulations...")
    win_counts = {p: 0 for p in players}

    for sim in range(N_SIMULATIONS):

        # Shuffle to randomise bracket position each simulation
        remaining = players.copy()
        np.random.shuffle(remaining)

        # Simulate rounds: 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
        while len(remaining) > 1:
            next_round = []

            for i in range(0, len(remaining) - 1, 2):
                p1      = remaining[i]
                p2      = remaining[i + 1]

                # Instant lookup — no model call
                prob_p1 = table.get((p1, p2), 0.5)
                winner  = p1 if np.random.random() < prob_p1 else p2
                next_round.append(winner)

            # Bye if odd number of players in this round
            if len(remaining) % 2 == 1:
                next_round.append(remaining[-1])

            remaining = next_round

        # Record the champion for this simulation
        win_counts[remaining[0]] += 1

        if (sim + 1) % 2000 == 0:
            print(f"  {sim + 1:,} / {N_SIMULATIONS:,} simulations done...")

    # Convert counts to probabilities
    win_probs = {
        p: win_counts[p] / N_SIMULATIONS
        for p in players
    }
    return dict(sorted(win_probs.items(), key=lambda x: -x[1]))


# ── Build 2026 predictions ────────────────────────────────────────────────────

def build_predictions(snapshot, model):
    print("\n── Building 2026 French Open Draw (128 players) ──")

    # Automatically pick top DRAW_SIZE active players by clay Elo
    draw_players = (
        snapshot
        .sort_values("clay_elo", ascending=False)
        .head(DRAW_SIZE)["player_name"]
        .tolist()
    )

    print(f"  Top 5 seeds in draw:")
    for i, p in enumerate(draw_players[:5], 1):
        row = snapshot[snapshot["player_name"] == p].iloc[0]
        print(f"    {i}. {p}  (clay Elo: {row['clay_elo']:.0f})")

    # Run fast Monte Carlo simulation
    win_probs = monte_carlo_simulation(draw_players, snapshot, model)

    # Build results dataframe
    rows = []
    for rank, (player, prob) in enumerate(win_probs.items(), 1):
        snap = snapshot[snapshot["player_name"] == player]
        if not snap.empty:
            s = snap.iloc[0]
            rows.append({
                "rank"            : rank,
                "player"          : player,
                "win_probability" : prob,
                "clay_elo"        : round(s["clay_elo"]),
                "clay_wr_pct"     : round(s["clay_wr"] * 100, 1),
                "form_pct"        : round(s["form"] * 100, 1),
            })

    return pd.DataFrame(rows)


# ── Print predictions table ───────────────────────────────────────────────────

def print_predictions(predictions):
    top = predictions.head(DISPLAY_TOP)

    print("\n" + "="*68)
    print("  2026 FRENCH OPEN - PREDICTED WIN PROBABILITIES")
    print("  Carlos Alcaraz excluded (wrist injury)")
    print(f"  Top {DISPLAY_TOP} of {DRAW_SIZE} players shown")
    print("="*68)
    print(f"  {'Rank':<6} {'Player':<26} {'Win Prob':>9} "
          f"{'Clay Elo':>9} {'Clay WR':>8} {'Form':>6}")
    print(f"  {'-'*65}")

    for _, row in top.iterrows():
        print(f"  {int(row['rank']):<6} {row['player']:<26} "
              f"{row['win_probability']:>8.1%}  "
              f"{int(row['clay_elo']):>8}  "
              f"{row['clay_wr_pct']:>6.1f}%  "
              f"{row['form_pct']:>4.1f}%")

    print("="*68)
    print(f"\n  Predicted champion : {predictions.iloc[0]['player']}")
    print(f"  Win probability    : {predictions.iloc[0]['win_probability']:.1%}")
    print(f"  Runner-up          : {predictions.iloc[1]['player']}")
    print(f"  Runner-up prob     : {predictions.iloc[1]['win_probability']:.1%}")


# ── Plot win probability chart ────────────────────────────────────────────────

def plot_predictions(predictions, output_path):
    top    = predictions.head(15)
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#1D9E75" if i == 0 else "#B4B2A9"
              for i in range(len(top))]

    bars = ax.barh(
        top["player"][::-1],
        top["win_probability"][::-1],
        color=colors[::-1]
    )

    ax.set_xlabel("Win probability", fontsize=11)
    ax.set_title(
        "2026 French Open - Predicted Win Probabilities\n"
        "(Carlos Alcaraz excluded - wrist injury)",
        fontsize=12, fontweight="bold"
    )
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0%}")
    )
    ax.spines[["top", "right"]].set_visible(False)

    for bar, val in zip(bars[::-1], top["win_probability"]):
        ax.text(
            val + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}", va="center", fontsize=9
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {output_path}")


# ── Save model and outputs ────────────────────────────────────────────────────

def save_model(xgb_model, lr_model, lr_scaler):
    model_path = MODEL_DIR / "xgboost_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(xgb_model, f)
    print(f"Saved -> {model_path}")

    lr_path = MODEL_DIR / "logistic_model.pkl"
    with open(lr_path, "wb") as f:
        pickle.dump({"model": lr_model, "scaler": lr_scaler}, f)
    print(f"Saved -> {lr_path}")


def save_report(results, predictions, baseline_acc, output_path):
    lines = [
        "French Open 2026 - Model Training Report",
        "=" * 50,
        f"Draw size       : {DRAW_SIZE} players",
        f"Simulations     : {N_SIMULATIONS:,}",
        f"Baseline acc    : {baseline_acc:.1%}",
        "",
    ]
    for name, metrics in results.items():
        lines.append(f"{name}:")
        for k, v in metrics.items():
            lines.append(f"  {k:<14}: {v:.4f}")
        lines.append("")

    lines.append(f"Top {DISPLAY_TOP} Win Probabilities:")
    lines.append("-" * 40)
    for _, row in predictions.head(DISPLAY_TOP).iterrows():
        lines.append(
            f"  {int(row['rank']):<4} {row['player']:<26} "
            f"{row['win_probability']:.1%}"
        )

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved -> {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n── Step 3: Model Training (128-player draw, fast simulation) ──\n")

    # Load
    features = load_features()
    snapshot = load_snapshot()

    # Split by date
    X_train, y_train, X_test, y_test = split_data(features)

    # Baseline
    baseline_acc = evaluate_baseline(X_test, y_test)

    # Train
    xgb_model           = train_xgboost(X_train, y_train)
    lr_model, lr_scaler = train_logistic(X_train, y_train)

    # Evaluate
    results = evaluate_models(
        xgb_model, lr_model, lr_scaler,
        X_train, y_train, X_test, y_test, baseline_acc
    )

    # Feature importance chart
    print("\n── Plotting feature importance ──")
    plot_feature_importance(xgb_model, MODEL_DIR / "feature_importance.png")

    # 2026 predictions — fast simulation
    predictions = build_predictions(snapshot, xgb_model)
    print_predictions(predictions)

    # Win probability chart
    print("\n── Plotting win probabilities ──")
    plot_predictions(predictions, MODEL_DIR / "predictions_2026.png")

    # Save everything
    save_model(xgb_model, lr_model, lr_scaler)
    predictions.to_csv(DATA_DIR / "predictions_2026.csv", index=False)
    print(f"Saved -> data/predictions_2026.csv")

    save_report(results, predictions, baseline_acc, MODEL_DIR / "model_report.txt")

    print("\n Step 3 complete.")
    print("Next step: run  step4_evaluation.py")


if __name__ == "__main__":
    main()
