"""
French Open 2026 — Head-to-Head Match Predictor Website
========================================================
A standalone web app that predicts the winner between
any two players using the trained XGBoost model.

Run:
    streamlit run match_predictor.py

Requirements (already installed):
    streamlit, pandas, numpy, pickle, matplotlib
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR  = Path("data")
MODEL_DIR = Path("models")

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

EXCLUDED_PLAYERS = [
    "Carlos Alcaraz", "Rafael Nadal", "Roger Federer",
    "Andy Murray", "Juan Martin del Potro", "Robin Soderling",
    "Stan Wawrinka", "Marin Cilic", "Kei Nishikori",
    "Gael Monfils", "Tomas Berdych", "Milos Raonic", "Nick Kyrgios",
]

SURFACE_EMOJI = {"Clay": "🟤", "Hard": "🔵", "Grass": "🟢"}

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title   = "Tennis Match Predictor",
    page_icon    = "🎾",
    layout       = "centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .big-title {
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.1rem;
        text-align: center;
        color: #888780;
        margin-bottom: 2rem;
    }
    .winner-box {
        background: linear-gradient(135deg, #1D9E75, #0F6E56);
        color: white;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .winner-name {
        font-size: 2rem;
        font-weight: 700;
    }
    .winner-prob {
        font-size: 1.3rem;
        opacity: 0.9;
    }
    .stat-card {
        background: #F7F7F5;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        margin: 0.3rem 0;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #888780;
        margin-bottom: 0.2rem;
    }
    .stat-value {
        font-size: 1.3rem;
        font-weight: 600;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin: 1.5rem 0 0.5rem;
        border-left: 4px solid #1D9E75;
        padding-left: 0.7rem;
    }
    .tip-box {
        background: #EFF9F5;
        border: 1px solid #1D9E75;
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #085041;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model and data ───────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    path = MODEL_DIR / "xgboost_model.pkl"
    if not path.exists():
        st.error("Model not found. Run step3_model_training.py first.")
        st.stop()
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_snapshot():
    path = DATA_DIR / "player_snapshot.csv"
    if not path.exists():
        st.error("player_snapshot.csv not found. Run step2_feature_engineering.py first.")
        st.stop()
    df = pd.read_csv(path)
    df = df[~df["player_name"].isin(EXCLUDED_PLAYERS)].reset_index(drop=True)
    return df.sort_values("clay_elo", ascending=False).reset_index(drop=True)


model    = load_model()
snapshot = load_snapshot()
players  = snapshot["player_name"].tolist()


# ── Prediction logic ──────────────────────────────────────────────────────────

def get_player_stats(name):
    """Fetch a player's stats from the snapshot."""
    row = snapshot[snapshot["player_name"] == name]
    if row.empty:
        return None
    return row.iloc[0]


def predict_match(p1_name, p2_name):
    """
    Predict win probability for p1 vs p2 on clay.
    Returns dict with probabilities and feature breakdown.
    """
    s1 = get_player_stats(p1_name)
    s2 = get_player_stats(p2_name)

    if s1 is None or s2 is None:
        return None

    features = {
        "diff_elo"         : float(s1["elo"])      - float(s2["elo"]),
        "diff_clay_elo"    : float(s1["clay_elo"]) - float(s2["clay_elo"]),
        "diff_form"        : float(s1["form"])     - float(s2["form"]),
        "diff_clay_wr"     : float(s1["clay_wr"])  - float(s2["clay_wr"]),
        "diff_h2h_clay"    : 0.0,
        "diff_h2h_overall" : 0.0,
        "diff_rg_best"     : float(s1.get("rg_best", 0)) -
                             float(s2.get("rg_best", 0)),
        "diff_fatigue"     : 0.0,
        "diff_rank"        : (float(s2.get("rank", 200) or 200)) -
                             (float(s1.get("rank", 200) or 200)),
    }

    X       = pd.DataFrame([features])[FEATURE_COLS]
    prob_p1 = float(model.predict_proba(X)[0][1])
    prob_p2 = 1.0 - prob_p1

    return {
        "prob_p1"   : prob_p1,
        "prob_p2"   : prob_p2,
        "winner"    : p1_name if prob_p1 > prob_p2 else p2_name,
        "features"  : features,
        "s1"        : s1,
        "s2"        : s2,
    }


def advantage_label(diff, p1, p2, fmt="raw"):
    """Return human-readable advantage string."""
    if abs(diff) < 0.01:
        return "Even", "—"
    favour = p1 if diff > 0 else p2
    if fmt == "pct":
        return favour, f"{abs(diff)*100:.1f}%"
    elif fmt == "elo":
        return favour, f"+{abs(diff):.0f} Elo"
    else:
        return favour, f"{diff:+.2f}"


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown('<div class="big-title">🎾 Tennis Match Predictor</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Predict the winner of any clay court match '
    'using our XGBoost model trained on 15 years of ATP data</div>',
    unsafe_allow_html=True
)

# ── Player selection ──────────────────────────────────────────────────────────

st.markdown('<div class="section-title">Select Players</div>',
            unsafe_allow_html=True)

col1, col_vs, col2 = st.columns([10, 1, 10])

with col1:
    p1 = st.selectbox(
        "Player 1",
        options      = players,
        index        = 0,
        label_visibility = "collapsed",
        placeholder  = "Select Player 1",
    )
    # Show player quick stats
    s1 = get_player_stats(p1)
    if s1 is not None:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Clay Elo</div>
            <div class="stat-value">{s1['clay_elo']:.0f}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Clay Win Rate</div>
            <div class="stat-value">{s1['clay_wr']*100:.1f}%</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Current Form</div>
            <div class="stat-value">{s1['form']*100:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)

with col_vs:
    st.markdown("<br><br><br><center><b>vs</b></center>",
                unsafe_allow_html=True)

with col2:
    p2 = st.selectbox(
        "Player 2",
        options      = players,
        index        = 1,
        label_visibility = "collapsed",
        placeholder  = "Select Player 2",
    )
    s2 = get_player_stats(p2)
    if s2 is not None:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Clay Elo</div>
            <div class="stat-value">{s2['clay_elo']:.0f}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Clay Win Rate</div>
            <div class="stat-value">{s2['clay_wr']*100:.1f}%</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Current Form</div>
            <div class="stat-value">{s2['form']*100:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)

# ── Predict button ────────────────────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button(
    "🎾 Predict Match Winner",
    use_container_width = True,
    type                = "primary",
)

# ── Results ───────────────────────────────────────────────────────────────────

if predict_btn:
    if p1 == p2:
        st.warning("Please select two different players.")
    else:
        result = predict_match(p1, p2)

        if result is None:
            st.error("Could not find player data. Try different players.")
        else:
            prob_p1 = result["prob_p1"]
            prob_p2 = result["prob_p2"]
            winner  = result["winner"]
            loser   = p2 if winner == p1 else p1
            w_prob  = max(prob_p1, prob_p2)
            l_prob  = min(prob_p1, prob_p2)

            st.markdown("---")

            # ── Winner announcement ──
            st.markdown(f"""
            <div class="winner-box">
                <div style="font-size:1rem;opacity:0.8;margin-bottom:0.5rem">
                    🏆 Predicted Winner
                </div>
                <div class="winner-name">{winner}</div>
                <div class="winner-prob">{w_prob:.1%} win probability</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Probability bar ──
            fig, ax = plt.subplots(figsize=(8, 1.2))
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")

            ax.barh([0], [prob_p1], color="#1D9E75", height=0.6)
            ax.barh([0], [prob_p2], left=[prob_p1],
                    color="#D85A30", height=0.6)

            ax.text(
                prob_p1 / 2, 0,
                f"{p1}  {prob_p1:.1%}",
                ha="center", va="center",
                fontsize=11, color="white", fontweight="bold"
            )
            ax.text(
                prob_p1 + prob_p2 / 2, 0,
                f"{p2}  {prob_p2:.1%}",
                ha="center", va="center",
                fontsize=11, color="white", fontweight="bold"
            )

            ax.set_xlim(0, 1)
            ax.axis("off")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            # ── Confidence level ──
            margin = abs(prob_p1 - prob_p2)
            if margin > 0.3:
                confidence = "🟢 High confidence"
                conf_desc  = "The model is strongly confident in this prediction."
            elif margin > 0.15:
                confidence = "🟡 Medium confidence"
                conf_desc  = "The model leans toward this outcome but an upset is possible."
            else:
                confidence = "🔴 Low confidence — toss-up"
                conf_desc  = "This is very close. Either player could realistically win."

            st.markdown(f"""
            <div class="tip-box">
                <b>{confidence}</b><br>{conf_desc}
            </div>
            """, unsafe_allow_html=True)

            # ── Feature breakdown ──
            st.markdown(
                '<div class="section-title">What\'s driving the prediction?</div>',
                unsafe_allow_html=True
            )

            features     = result["features"]
            breakdown    = [
                ("Clay Elo rating",      features["diff_clay_elo"],  "elo"),
                ("General Elo rating",   features["diff_elo"],       "elo"),
                ("Recent form",          features["diff_form"],      "pct"),
                ("Clay win rate",        features["diff_clay_wr"],   "pct"),
                ("RG history",           features["diff_rg_best"],   "raw"),
                ("Rest advantage",       features["diff_fatigue"],   "raw"),
                ("ATP ranking",          features["diff_rank"],      "raw"),
            ]

            for label, diff, fmt in breakdown:
                favour, val_str = advantage_label(diff, p1, p2, fmt)
                col_a, col_b, col_c = st.columns([3, 2, 2])
                with col_a:
                    st.markdown(f"**{label}**")
                with col_b:
                    color = (
                        "#1D9E75" if favour == p1
                        else "#D85A30" if favour == p2
                        else "#888780"
                    )
                    st.markdown(
                        f"<span style='color:{color};font-weight:500'>"
                        f"favours {favour}</span>",
                        unsafe_allow_html=True
                    )
                with col_c:
                    st.markdown(f"`{val_str}`")

            # ── Player comparison table ──
            st.markdown(
                '<div class="section-title">Player stats comparison</div>',
                unsafe_allow_html=True
            )

            s1 = result["s1"]
            s2 = result["s2"]

            comp_df = pd.DataFrame({
                "Stat"    : [
                    "Clay Elo", "General Elo",
                    "Clay Win Rate", "Recent Form",
                    "RG Best Round",
                ],
                p1        : [
                    f"{s1['clay_elo']:.0f}",
                    f"{s1['elo']:.0f}",
                    f"{s1['clay_wr']*100:.1f}%",
                    f"{s1['form']*100:.0f}%",
                    int(s1.get("rg_best", 0)),
                ],
                p2        : [
                    f"{s2['clay_elo']:.0f}",
                    f"{s2['elo']:.0f}",
                    f"{s2['clay_wr']*100:.1f}%",
                    f"{s2['form']*100:.0f}%",
                    int(s2.get("rg_best", 0)),
                ],
                "Edge"    : [
                    p1 if s1["clay_elo"] > s2["clay_elo"] else p2,
                    p1 if s1["elo"]      > s2["elo"]      else p2,
                    p1 if s1["clay_wr"]  > s2["clay_wr"]  else p2,
                    p1 if s1["form"]     > s2["form"]      else p2,
                    p1 if s1.get("rg_best", 0) >
                          s2.get("rg_best", 0)             else p2,
                ],
            })

            st.dataframe(
                comp_df,
                use_container_width = True,
                hide_index          = True,
            )

            # ── Simulate 5 match outcomes ──
            st.markdown(
                '<div class="section-title">'
                'What could happen in 5 simulated matches?'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                "_Each simulated match randomly samples the outcome "
                "using the predicted probabilities:_"
            )

            np.random.seed(None)  # fresh randomness each run
            outcomes = []
            p1_wins  = 0
            p2_wins  = 0

            for i in range(5):
                won = np.random.random() < prob_p1
                winner_sim = p1 if won else p2
                loser_sim  = p2 if won else p1
                if won:
                    p1_wins += 1
                else:
                    p2_wins += 1
                outcomes.append(
                    f"Match {i+1}: **{winner_sim}** def. {loser_sim}"
                )

            for outcome in outcomes:
                st.markdown(f"- {outcome}")

            st.markdown(
                f"\n_In this simulation: "
                f"{p1} won {p1_wins}/5, "
                f"{p2} won {p2_wins}/5_"
            )

            # ── Disclaimer ──
            st.markdown("---")
            st.caption(
                "⚠️ This prediction is based on historical ATP clay court data "
                "(2010–2025). It does not account for live match conditions, "
                "injuries, or real-time form changes. For informational and "
                "educational purposes only."
            )

# ── Footer when no prediction yet ────────────────────────────────────────────

else:
    st.markdown("---")
    st.markdown(
        '<div class="section-title">How it works</div>',
        unsafe_allow_html=True
    )

    col_h1, col_h2, col_h3 = st.columns(3)
    with col_h1:
        st.markdown("""
        **1. Select players**
        Choose any two active ATP players from the dropdown lists above.
        """)
    with col_h2:
        st.markdown("""
        **2. Model predicts**
        Our XGBoost model compares their clay Elo, form, win rates
        and RG history to calculate win probability.
        """)
    with col_h3:
        st.markdown("""
        **3. See the breakdown**
        View win probability, confidence level, feature-by-feature
        breakdown and a stats comparison table.
        """)

    st.markdown("---")
    st.markdown(
        '<div class="section-title">Model details</div>',
        unsafe_allow_html=True
    )

    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    with col_d1:
        st.metric("Algorithm",     "XGBoost")
    with col_d2:
        st.metric("Test Accuracy", "74.6%")
    with col_d3:
        st.metric("ROC-AUC",       "0.8122")
    with col_d4:
        st.metric("Training data", "15 years")

    st.markdown("---")
    st.caption(
        "Built with Python · XGBoost · Streamlit · "
        "Data: JeffSackmann/tennis_atp (2010–2025)"
    )
