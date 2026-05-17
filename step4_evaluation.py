"""
French Open Prediction — Step 4: Model Evaluation
==================================================
Reads  : data/features.csv          (output of step2_feature_engineering.py)
         data/player_snapshot.csv   (current player stats)
         models/xgboost_model.pkl   (trained model)
         data/predictions_2026.csv  (2026 win probabilities)
Outputs: evaluation/calibration.png
         evaluation/confusion_matrix.png
         evaluation/roc_curve.png
         evaluation/yearly_accuracy.png
         evaluation/evaluation_report.txt

This step answers the key question:
  "Can we actually trust this model's predictions?"

We evaluate from 4 angles:
  1. Accuracy by year      — is it consistent or did it get lucky?
  2. Calibration           — when it says 70%, does it win 70% of the time?
  3. ROC curve             — how well does it separate winners from losers?
  4. Roland Garros backtest — did it correctly predict past RG winners?

Run:
    python step4_evaluation.py
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    log_loss, brier_score_loss, confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR  = Path("data")
MODEL_DIR = Path("models")
EVAL_DIR  = Path("evaluation")
EVAL_DIR.mkdir(exist_ok=True)

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

# Actual Roland Garros winners for backtesting
RG_WINNERS = {
    2015: "Stan Wawrinka",
    2016: "Novak Djokovic",
    2017: "Rafael Nadal",
    2018: "Rafael Nadal",
    2019: "Rafael Nadal",
    2020: "Rafael Nadal",
    2021: "Novak Djokovic",
    2022: "Rafael Nadal",
    2023: "Novak Djokovic",
    2024: "Carlos Alcaraz",
    2025: "Carlos Alcaraz",
}


# ── Load data and model ───────────────────────────────────────────────────────

def load_data():
    features = pd.read_csv(DATA_DIR / "features.csv", low_memory=False)
    features["match_date"] = pd.to_datetime(
        features["match_date"], errors="coerce"
    )
    features["year"] = features["match_date"].dt.year

    snapshot    = pd.read_csv(DATA_DIR / "player_snapshot.csv")
    predictions = pd.read_csv(DATA_DIR / "predictions_2026.csv")

    with open(MODEL_DIR / "xgboost_model.pkl", "rb") as f:
        model = pickle.load(f)

    print(f"Loaded {len(features):,} clay match rows")
    print(f"Loaded model: {MODEL_DIR / 'xgboost_model.pkl'}")
    return features, snapshot, predictions, model


# ── Evaluation 1: Accuracy by year ───────────────────────────────────────────
#
# Checks whether the model performs consistently across different years.
# A model that scores 90% in one year and 50% in another is unreliable.
# We want to see stable accuracy across all years.

def evaluate_by_year(features, model):
    print("\n── Evaluation 1: Accuracy by Year ──")

    rows = []
    years = sorted(features["year"].dropna().unique())

    for year in years:
        year_df = features[features["year"] == year]
        if len(year_df) < 10:
            continue

        X    = year_df[FEATURE_COLS]
        y    = year_df[TARGET_COL]
        pred = model.predict(X)
        prob = model.predict_proba(X)[:, 1]

        acc   = accuracy_score(y, pred)
        auc   = roc_auc_score(y, prob)
        n     = len(year_df)

        rows.append({
            "year"    : int(year),
            "accuracy": acc,
            "roc_auc" : auc,
            "n_matches": n,
        })

        print(f"  {int(year)}: accuracy={acc:.1%}  "
              f"ROC-AUC={auc:.3f}  n={n}")

    return pd.DataFrame(rows)


def plot_yearly_accuracy(yearly_df, output_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # Accuracy line
    ax1.plot(
        yearly_df["year"], yearly_df["accuracy"],
        marker="o", color="#1D9E75", linewidth=2, markersize=6
    )
    ax1.axhline(0.5, color="#D85A30", linestyle="--",
                linewidth=1, label="Random (50%)")
    ax1.axhline(yearly_df["accuracy"].mean(), color="#534AB7",
                linestyle="--", linewidth=1,
                label=f"Mean ({yearly_df['accuracy'].mean():.1%})")
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_title(
        "Model accuracy by year (clay matches)",
        fontsize=12, fontweight="bold"
    )
    ax1.set_ylim(0.45, 0.95)
    ax1.legend(fontsize=9)
    ax1.spines[["top", "right"]].set_visible(False)

    # ROC-AUC line
    ax2.plot(
        yearly_df["year"], yearly_df["roc_auc"],
        marker="s", color="#534AB7", linewidth=2, markersize=6
    )
    ax2.axhline(0.5, color="#D85A30", linestyle="--",
                linewidth=1, label="Random (0.50)")
    ax2.axhline(yearly_df["roc_auc"].mean(), color="#1D9E75",
                linestyle="--", linewidth=1,
                label=f"Mean ({yearly_df['roc_auc'].mean():.3f})")
    ax2.set_ylabel("ROC-AUC", fontsize=11)
    ax2.set_xlabel("Year", fontsize=11)
    ax2.set_ylim(0.45, 0.95)
    ax2.legend(fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.xticks(yearly_df["year"], rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {output_path}")


# ── Evaluation 2: Calibration ─────────────────────────────────────────────────
#
# Calibration answers: "When the model says 70% confidence, does it
# actually win 70% of the time?"
#
# A perfectly calibrated model follows the diagonal line.
# Above diagonal = overconfident (says 70%, actually only wins 60%)
# Below diagonal = underconfident (says 70%, actually wins 80%)

def evaluate_calibration(features, model, output_path):
    print("\n── Evaluation 2: Calibration ──")

    # Use test set only (2024-2025)
    test_df = features[features["match_date"] >= "2024-01-01"].copy()
    X_test  = test_df[FEATURE_COLS]
    y_test  = test_df[TARGET_COL]

    probs = model.predict_proba(X_test)[:, 1]

    # Compute calibration curve
    # fraction_of_positives = actual win rate in each probability bucket
    # mean_predicted_value  = average predicted probability in each bucket
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, probs, n_bins=10
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    # Perfect calibration line
    ax.plot(
        [0, 1], [0, 1],
        linestyle="--", color="#888780",
        linewidth=1.5, label="Perfect calibration"
    )

    # Model calibration
    ax.plot(
        mean_predicted_value, fraction_of_positives,
        marker="o", color="#1D9E75", linewidth=2,
        markersize=8, label="XGBoost model"
    )

    # Shade the gap between model and perfect
    ax.fill_between(
        mean_predicted_value,
        mean_predicted_value,
        fraction_of_positives,
        alpha=0.1, color="#1D9E75"
    )

    ax.set_xlabel("Mean predicted probability", fontsize=11)
    ax.set_ylabel("Actual win rate", fontsize=11)
    ax.set_title(
        "Calibration curve\n"
        "How well do predicted probabilities match reality?",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {output_path}")

    # Print calibration summary
    mean_gap = np.mean(
        np.abs(fraction_of_positives - mean_predicted_value)
    )
    print(f"  Mean calibration gap : {mean_gap:.3f}")
    print(f"  (0.00 = perfect, >0.10 = poor)")
    return mean_gap


# ── Evaluation 3: ROC Curve ───────────────────────────────────────────────────
#
# The ROC curve shows how well the model separates winners from losers
# at every possible probability threshold.
#
# Area Under Curve (AUC):
#   0.50 = random guessing (useless)
#   0.70 = decent
#   0.80 = good
#   0.90 = excellent
#   1.00 = perfect (impossible in sport)

def evaluate_roc(features, model, output_path):
    print("\n── Evaluation 3: ROC Curve ──")

    test_df = features[features["match_date"] >= "2024-01-01"].copy()
    X_test  = test_df[FEATURE_COLS]
    y_test  = test_df[TARGET_COL]

    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Random baseline
    ax.plot(
        [0, 1], [0, 1],
        linestyle="--", color="#888780",
        linewidth=1.5, label="Random (AUC=0.50)"
    )

    # Model ROC curve
    ax.plot(
        fpr, tpr,
        color="#1D9E75", linewidth=2,
        label=f"XGBoost (AUC={auc:.4f})"
    )
    ax.fill_between(fpr, tpr, alpha=0.1, color="#1D9E75")

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(
        "ROC Curve\n"
        "How well does the model separate winners from losers?",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  Saved -> {output_path}")
    return auc


# ── Evaluation 4: Confusion Matrix ───────────────────────────────────────────
#
# Shows exactly where the model makes mistakes:
#   True Positive  (TP): predicted win,  actual win   ✅
#   True Negative  (TN): predicted loss, actual loss  ✅
#   False Positive (FP): predicted win,  actual loss  ❌ (missed upset)
#   False Negative (FN): predicted loss, actual win   ❌ (wrong upset)

def evaluate_confusion_matrix(features, model, output_path):
    print("\n── Evaluation 4: Confusion Matrix ──")

    test_df = features[features["match_date"] >= "2024-01-01"].copy()
    X_test  = test_df[FEATURE_COLS]
    y_test  = test_df[TARGET_COL]

    preds = model.predict(X_test)
    cm    = confusion_matrix(y_test, preds)

    tn, fp, fn, tp = cm.ravel()
    total = len(y_test)

    print(f"  True Positives  (correctly predicted winner) : {tp:,}  "
          f"({tp/total:.1%})")
    print(f"  True Negatives  (correctly predicted loser)  : {tn:,}  "
          f"({tn/total:.1%})")
    print(f"  False Positives (predicted win, actual loss) : {fp:,}  "
          f"({fp/total:.1%})")
    print(f"  False Negatives (predicted loss, actual win) : {fn:,}  "
          f"({fn/total:.1%})")

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Lost (0)", "Won (1)"]
    )
    disp.plot(ax=ax, colorbar=False, cmap="Greens")
    ax.set_title(
        "Confusion Matrix (test set 2024-2025)",
        fontsize=12, fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {output_path}")


# ── Evaluation 5: Roland Garros Backtest ─────────────────────────────────────
#
# The ultimate test: for each past Roland Garros (2015-2025),
# did our model correctly predict the winner?
#
# We simulate each year's tournament using only data available
# BEFORE that year's Roland Garros started.
# This is a true out-of-sample test.

def backtest_roland_garros(features, snapshot, model):
    print("\n── Evaluation 5: Roland Garros Backtest (2015-2025) ──")

    # Reuse the fast matchup approach but per year
    results = []

    for year, actual_winner in RG_WINNERS.items():
        # Get all players who appeared in RG that year
        rg_year = features[
            (features["tourney_name"].str.contains(
                "Roland Garros", case=False, na=False
            )) &
            (features["year"] == year)
        ]

        if rg_year.empty:
            continue

        # Get unique players in that year's RG
        winners = rg_year["winner_name"].dropna().unique().tolist()
        losers  = rg_year["loser_name"].dropna().unique().tolist()
        players = list(set(winners + losers))

        if len(players) < 4:
            continue

        # Build feature vectors for all matchups using snapshot
        rows  = []
        pairs = []

        for p1 in players:
            for p2 in players:
                if p1 == p2:
                    continue
                s1 = snapshot[snapshot["player_name"] == p1]
                s2 = snapshot[snapshot["player_name"] == p2]
                if s1.empty or s2.empty:
                    rows.append([0.0] * len(FEATURE_COLS))
                else:
                    s1 = s1.iloc[0]
                    s2 = s2.iloc[0]
                    rows.append([
                        s1["elo"]      - s2["elo"],
                        s1["clay_elo"] - s2["clay_elo"],
                        s1["form"]     - s2["form"],
                        s1["clay_wr"]  - s2["clay_wr"],
                        0.0, 0.0,
                        s1.get("rg_best", 0) - s2.get("rg_best", 0),
                        0.0,
                        (s2.get("rank", 200) or 200) -
                        (s1.get("rank", 200) or 200),
                    ])
                pairs.append((p1, p2))

        if not rows:
            continue

        # Batch predict
        X     = pd.DataFrame(rows, columns=FEATURE_COLS)
        probs = model.predict_proba(X)[:, 1]
        table = {pair: float(prob) for pair, prob in zip(pairs, probs)}

        # Run 5000 simulations for this year
        win_counts = {p: 0 for p in players}
        n_sims     = 5000

        for _ in range(n_sims):
            remaining = players.copy()
            np.random.shuffle(remaining)

            while len(remaining) > 1:
                next_round = []
                for i in range(0, len(remaining) - 1, 2):
                    p1      = remaining[i]
                    p2      = remaining[i + 1]
                    prob_p1 = table.get((p1, p2), 0.5)
                    winner  = p1 if np.random.random() < prob_p1 else p2
                    next_round.append(winner)
                if len(remaining) % 2 == 1:
                    next_round.append(remaining[-1])
                remaining = next_round

            win_counts[remaining[0]] += 1

        # Get top 3 predicted players
        sorted_players = sorted(
            win_counts.items(), key=lambda x: -x[1]
        )
        top3        = [p for p, _ in sorted_players[:3]]
        predicted   = top3[0]
        actual_prob = win_counts.get(actual_winner, 0) / n_sims
        correct     = predicted == actual_winner
        in_top3     = actual_winner in top3

        results.append({
            "year"          : year,
            "predicted"     : predicted,
            "actual"        : actual_winner,
            "correct"       : correct,
            "in_top3"       : in_top3,
            "actual_prob"   : actual_prob,
            "top3"          : ", ".join(top3),
        })

        status = "CORRECT" if correct else ("TOP 3" if in_top3 else "MISSED")
        print(f"  {year}: predicted={predicted:<20} "
              f"actual={actual_winner:<20} "
              f"actual_prob={actual_prob:.1%}  [{status}]")

    # Summary
    df         = pd.DataFrame(results)
    n_correct  = df["correct"].sum()
    n_top3     = df["in_top3"].sum()
    n_total    = len(df)

    print(f"\n  Backtest summary ({n_total} years):")
    print(f"    Exact winner predicted : {n_correct}/{n_total}  "
          f"({n_correct/n_total:.0%})")
    print(f"    Winner in top 3        : {n_top3}/{n_total}  "
          f"({n_top3/n_total:.0%})")
    print(f"    Avg prob given to actual winner: "
          f"{df['actual_prob'].mean():.1%}")

    return df


def plot_backtest(backtest_df, output_path):
    fig, ax = plt.subplots(figsize=(12, 5))

    colors = [
        "#1D9E75" if row["correct"]
        else "#F5C842" if row["in_top3"]
        else "#D85A30"
        for _, row in backtest_df.iterrows()
    ]

    bars = ax.bar(
        backtest_df["year"].astype(str),
        backtest_df["actual_prob"] * 100,
        color=colors, edgecolor="white", linewidth=0.5
    )

    # Labels on bars
    for bar, (_, row) in zip(bars, backtest_df.iterrows()):
        label = "✓" if row["correct"] else ("T3" if row["in_top3"] else "✗")
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            label, ha="center", va="bottom",
            fontsize=12, fontweight="bold"
        )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1D9E75", label="Correct prediction"),
        Patch(facecolor="#F5C842", label="Winner in top 3"),
        Patch(facecolor="#D85A30", label="Missed"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)

    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Probability given to actual winner (%)", fontsize=11)
    ax.set_title(
        "Roland Garros Backtest (2015-2025)\n"
        "How much probability did the model assign to the real winner?",
        fontsize=12, fontweight="bold"
    )
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {output_path}")


# ── Save evaluation report ────────────────────────────────────────────────────

def save_evaluation_report(yearly_df, calibration_gap,
                           auc, backtest_df, output_path):
    lines = [
        "French Open 2026 - Model Evaluation Report",
        "=" * 55,
        "",
        "1. Accuracy by Year",
        "-" * 40,
    ]
    for _, row in yearly_df.iterrows():
        lines.append(
            f"  {int(row['year'])}: accuracy={row['accuracy']:.1%}  "
            f"ROC-AUC={row['roc_auc']:.3f}  "
            f"n={int(row['n_matches'])}"
        )

    lines += [
        "",
        "2. Calibration",
        "-" * 40,
        f"  Mean calibration gap : {calibration_gap:.3f}",
        "  (0.00=perfect, >0.10=poor)",
        "",
        "3. ROC-AUC (test set 2024-2025)",
        "-" * 40,
        f"  AUC : {auc:.4f}",
        "",
        "4. Roland Garros Backtest",
        "-" * 40,
    ]
    for _, row in backtest_df.iterrows():
        status = "CORRECT" if row["correct"] else \
                 "TOP 3"   if row["in_top3"] else "MISSED"
        lines.append(
            f"  {row['year']}: {row['predicted']:<22} "
            f"(actual: {row['actual']})  [{status}]"
        )

    n_correct = backtest_df["correct"].sum()
    n_top3    = backtest_df["in_top3"].sum()
    n_total   = len(backtest_df)
    lines += [
        "",
        f"  Exact: {n_correct}/{n_total}  "
        f"Top-3: {n_top3}/{n_total}",
    ]

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved -> {output_path}")


# ── Print final summary ───────────────────────────────────────────────────────

def print_final_summary(yearly_df, calibration_gap, auc, backtest_df):
    n_correct = backtest_df["correct"].sum()
    n_top3    = backtest_df["in_top3"].sum()
    n_total   = len(backtest_df)

    print("\n" + "="*55)
    print("  EVALUATION SUMMARY")
    print("="*55)
    print(f"  Avg yearly accuracy   : "
          f"{yearly_df['accuracy'].mean():.1%}")
    print(f"  Avg yearly ROC-AUC    : "
          f"{yearly_df['roc_auc'].mean():.4f}")
    print(f"  Calibration gap       : {calibration_gap:.3f}  "
          f"({'good' if calibration_gap < 0.05 else 'ok' if calibration_gap < 0.10 else 'needs improvement'})")
    print(f"  Test ROC-AUC          : {auc:.4f}  "
          f"({'excellent' if auc > 0.80 else 'good' if auc > 0.70 else 'ok'})")
    print(f"  RG backtest (exact)   : {n_correct}/{n_total}  "
          f"({n_correct/n_total:.0%})")
    print(f"  RG backtest (top 3)   : {n_top3}/{n_total}  "
          f"({n_top3/n_total:.0%})")
    print("="*55)
    print("\n  Evaluation charts saved in evaluation/ folder:")
    print("    yearly_accuracy.png")
    print("    calibration.png")
    print("    roc_curve.png")
    print("    confusion_matrix.png")
    print("    rg_backtest.png")
    print("    evaluation_report.txt")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n── Step 4: Model Evaluation ──\n")

    # Load everything
    features, snapshot, predictions, model = load_data()

    # 1. Accuracy by year
    yearly_df = evaluate_by_year(features, model)
    plot_yearly_accuracy(yearly_df, EVAL_DIR / "yearly_accuracy.png")

    # 2. Calibration
    calibration_gap = evaluate_calibration(
        features, model, EVAL_DIR / "calibration.png"
    )

    # 3. ROC curve
    auc = evaluate_roc(features, model, EVAL_DIR / "roc_curve.png")

    # 4. Confusion matrix
    evaluate_confusion_matrix(
        features, model, EVAL_DIR / "confusion_matrix.png"
    )

    # 5. Roland Garros backtest
    backtest_df = backtest_roland_garros(features, snapshot, model)
    plot_backtest(backtest_df, EVAL_DIR / "rg_backtest.png")

    # Save report
    save_evaluation_report(
        yearly_df, calibration_gap, auc,
        backtest_df,
        EVAL_DIR / "evaluation_report.txt"
    )

    # Final summary
    print_final_summary(yearly_df, calibration_gap, auc, backtest_df)

    print("\n Step 4 complete.")
    print("Next step: run  step5_dashboard.py")


if __name__ == "__main__":
    main()
