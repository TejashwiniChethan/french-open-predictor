"""
French Open Prediction — Step 5: Streamlit Dashboard
=====================================================
Reads  : data/predictions_2026.csv
         data/player_snapshot.csv
         models/xgboost_model.pkl
         models/feature_importance.png
         evaluation/evaluation_report.txt

Run:
    streamlit run step5_dashboard.py

This opens a web app in your browser showing:
  - 2026 French Open win probability table
  - Win probability bar chart
  - Head-to-head match predictor
  - Feature importance chart
  - Model evaluation summary
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
EVAL_DIR  = Path("evaluation")

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
    "Carlos Alcaraz",
    "Rafael Nadal",
    "Roger Federer",
    "Andy Murray",
    "Juan Martin del Potro",
    "Robin Soderling",
    "Stan Wawrinka",
    "Marin Cilic",
    "Kei Nishikori",
    "Gael Monfils",
    "Tomas Berdych",
    "Milos Raonic",
    "Nick Kyrgios",
]

# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title  = "French Open 2026 Predictor",
    page_icon   = "🎾",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Load data ─────────────────────────────────────────────────────────────────

@st.cache_data
def load_predictions():
    return pd.read_csv(DATA_DIR / "predictions_2026.csv")


@st.cache_data
def load_snapshot():
    df = pd.read_csv(DATA_DIR / "player_snapshot.csv")
    return df[~df["player_name"].isin(EXCLUDED_PLAYERS)].reset_index(drop=True)


@st.cache_resource
def load_model():
    with open(MODEL_DIR / "xgboost_model.pkl", "rb") as f:
        return pickle.load(f)


predictions = load_predictions()
snapshot    = load_snapshot()
model       = load_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/5/5c/Roland_Garros_Logo.svg/200px-Roland_Garros_Logo.svg.png", width=120)
    st.title("🎾 French Open 2026")
    st.markdown("**ML-powered tournament predictor**")
    st.markdown("---")

    st.markdown("### Model info")
    st.markdown("- **Algorithm**: XGBoost")
    st.markdown("- **Training data**: 2010–2023 ATP clay matches")
    st.markdown("- **Test accuracy**: 74.6%")
    st.markdown("- **ROC-AUC**: 0.8122")
    st.markdown("- **Calibration gap**: 0.020 (excellent)")
    st.markdown("---")

    st.markdown("### Note")
    st.warning(
        "Carlos Alcaraz excluded due to wrist injury. "
        "Jannik Sinner is the predicted favourite."
    )

    st.markdown("---")
    st.markdown("Built with Python · XGBoost · Streamlit")
    st.markdown("Data: JeffSackmann/tennis_atp")


# ── Header ────────────────────────────────────────────────────────────────────

st.title("🎾 2026 French Open Winner Predictor")
st.markdown(
    "Predicting the 2026 Roland Garros champion using machine learning — "
    "trained on 15 years of ATP clay court match data."
)

# ── Top metric cards ──────────────────────────────────────────────────────────

top3 = predictions.head(3)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label    = "🏆 Predicted Champion",
        value    = top3.iloc[0]["player"],
        delta    = f"{top3.iloc[0]['win_probability']:.1%} win probability",
    )
with col2:
    st.metric(
        label    = "🥈 Runner-up",
        value    = top3.iloc[1]["player"],
        delta    = f"{top3.iloc[1]['win_probability']:.1%} win probability",
    )
with col3:
    st.metric(
        label    = "🥉 Third Favourite",
        value    = top3.iloc[2]["player"],
        delta    = f"{top3.iloc[2]['win_probability']:.1%} win probability",
    )
with col4:
    st.metric(
        label    = "🎾 Draw Size",
        value    = "128 players",
        delta    = "10,000 simulations",
    )

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Win Probabilities",
    "⚔️  Head-to-Head",
    "📈 Model Evaluation",
    "🔍 Feature Importance",
])


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1: Win Probabilities
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.subheader("2026 French Open — Predicted Win Probabilities")
    st.markdown(
        "Probabilities calculated by simulating the 128-player bracket "
        "10,000 times using XGBoost match predictions."
    )

    col_left, col_right = st.columns([1, 1])

    # ── Probability table ──
    with col_left:
        st.markdown("#### Top 20 contenders")

        display_df = predictions.head(20).copy()
        display_df["win_probability"] = display_df["win_probability"].apply(
            lambda x: f"{x:.1%}"
        )
        display_df["clay_wr_pct"] = display_df["clay_wr_pct"].apply(
            lambda x: f"{x:.1f}%"
        )
        display_df["form_pct"] = display_df["form_pct"].apply(
            lambda x: f"{x:.1f}%"
        )
        display_df = display_df.rename(columns={
            "rank"            : "Rank",
            "player"          : "Player",
            "win_probability" : "Win Prob",
            "clay_elo"        : "Clay Elo",
            "clay_wr_pct"     : "Clay WR",
            "form_pct"        : "Form",
        })

        st.dataframe(
            display_df[["Rank", "Player", "Win Prob",
                         "Clay Elo", "Clay WR", "Form"]],
            use_container_width = True,
            hide_index          = True,
        )

    # ── Probability bar chart ──
    with col_right:
        st.markdown("#### Win probability chart")

        top15  = predictions.head(15)
        fig, ax = plt.subplots(figsize=(7, 6))

        colors = [
            "#1D9E75" if i == 0
            else "#534AB7" if i == 1
            else "#B4B2A9"
            for i in range(len(top15))
        ]

        bars = ax.barh(
            top15["player"][::-1],
            top15["win_probability"][::-1],
            color=colors[::-1],
        )

        ax.set_xlabel("Win probability", fontsize=10)
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.0%}")
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(
            "2026 French Open Win Probabilities\n(Alcaraz excluded)",
            fontsize=11, fontweight="bold"
        )

        for bar, val in zip(bars[::-1], top15["win_probability"]):
            ax.text(
                val + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=8
            )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Probability donut ──
    st.markdown("---")
    st.markdown("#### How the probability is distributed")

    top5_prob  = predictions.head(5)["win_probability"].sum()
    rest_prob  = 1 - top5_prob

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Top 2 combined",
                  f"{predictions.head(2)['win_probability'].sum():.1%}",
                  "Sinner + Djokovic")
    with col_b:
        st.metric("Top 5 combined",
                  f"{top5_prob:.1%}",
                  "Top 5 favourites")
    with col_c:
        st.metric("Field (others)",
                  f"{rest_prob:.1%}",
                  "Remaining 123 players")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2: Head-to-Head Predictor
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("⚔️ Head-to-Head Match Predictor")
    st.markdown(
        "Select any two players and see the predicted win probability "
        "for a clay court match between them."
    )

    # Get active players sorted by clay Elo
    active_players = (
        snapshot
        .sort_values("clay_elo", ascending=False)
        .head(64)["player_name"]
        .tolist()
    )

    col_p1, col_vs, col_p2 = st.columns([5, 1, 5])

    with col_p1:
        p1 = st.selectbox(
            "Player A",
            options  = active_players,
            index    = 0,
        )
    with col_vs:
        st.markdown("<br><br><center><b>vs</b></center>",
                    unsafe_allow_html=True)
    with col_p2:
        p2 = st.selectbox(
            "Player B",
            options  = active_players,
            index    = 1,
        )

    if p1 != p2:
        # Get snapshot rows
        s1 = snapshot[snapshot["player_name"] == p1].iloc[0]
        s2 = snapshot[snapshot["player_name"] == p2].iloc[0]

        # Build feature vector
        features_dict = {
            "diff_elo"         : s1["elo"]      - s2["elo"],
            "diff_clay_elo"    : s1["clay_elo"] - s2["clay_elo"],
            "diff_form"        : s1["form"]     - s2["form"],
            "diff_clay_wr"     : s1["clay_wr"]  - s2["clay_wr"],
            "diff_h2h_clay"    : 0.0,
            "diff_h2h_overall" : 0.0,
            "diff_rg_best"     : s1.get("rg_best", 0) - s2.get("rg_best", 0),
            "diff_fatigue"     : 0.0,
            "diff_rank"        : (s2.get("rank", 200) or 200) -
                                 (s1.get("rank", 200) or 200),
        }

        X        = pd.DataFrame([features_dict])[FEATURE_COLS]
        prob_p1  = float(model.predict_proba(X)[0][1])
        prob_p2  = 1 - prob_p1

        # Result display
        st.markdown("---")
        winner = p1 if prob_p1 > prob_p2 else p2
        st.markdown(f"### 🏆 Predicted winner: **{winner}**")

        col_r1, col_r2 = st.columns(2)

        with col_r1:
            st.metric(
                label = p1,
                value = f"{prob_p1:.1%}",
                delta = "win probability"
            )
        with col_r2:
            st.metric(
                label = p2,
                value = f"{prob_p2:.1%}",
                delta = "win probability"
            )

        # Probability bar
        fig, ax = plt.subplots(figsize=(8, 1.2))
        ax.barh([0], [prob_p1], color="#1D9E75", height=0.5)
        ax.barh([0], [prob_p2], left=[prob_p1], color="#D85A30", height=0.5)
        ax.set_xlim(0, 1)
        ax.axis("off")
        ax.text(prob_p1 / 2, 0, f"{p1}\n{prob_p1:.1%}",
                ha="center", va="center", fontsize=10,
                color="white", fontweight="bold")
        ax.text(prob_p1 + prob_p2 / 2, 0, f"{p2}\n{prob_p2:.1%}",
                ha="center", va="center", fontsize=10,
                color="white", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Feature breakdown
        st.markdown("---")
        st.markdown("#### What's driving this prediction?")

        feature_labels = {
            "diff_clay_elo"    : "Clay Elo rating",
            "diff_elo"         : "General Elo rating",
            "diff_form"        : "Recent form",
            "diff_clay_wr"     : "Clay win rate",
            "diff_rg_best"     : "Roland Garros history",
            "diff_rank"        : "ATP ranking",
        }

        col_feat1, col_feat2 = st.columns(2)
        items = list(feature_labels.items())

        for i, (feat, label) in enumerate(items):
            raw_val = features_dict[feat]
            col     = col_feat1 if i % 2 == 0 else col_feat2
            with col:
                if raw_val > 0.01:
                    direction = f"favours {p1}"
                    delta_str = f"+{raw_val:.2f}"
                elif raw_val < -0.01:
                    direction = f"favours {p2}"
                    delta_str = f"{raw_val:.2f}"
                else:
                    direction = "roughly equal"
                    delta_str = "0.00"
                st.metric(label=label, value=direction, delta=delta_str)

    else:
        st.warning("Please select two different players.")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3: Model Evaluation
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("📈 Model Evaluation")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Test Accuracy",    "74.6%",  "+10.6% vs baseline")
    with col_m2:
        st.metric("ROC-AUC",          "0.8122", "Excellent (>0.80)")
    with col_m3:
        st.metric("Calibration Gap",  "0.020",  "Good (<0.05)")
    with col_m4:
        st.metric("RG Backtest Top3", "9/11",   "82% success rate")

    st.markdown("---")

    col_ev1, col_ev2 = st.columns(2)

    with col_ev1:
        st.markdown("#### Yearly accuracy (2010–2025)")
        if (EVAL_DIR / "yearly_accuracy.png").exists():
            st.image(str(EVAL_DIR / "yearly_accuracy.png"),
                     use_container_width=True)

        st.markdown("#### ROC Curve")
        if (EVAL_DIR / "roc_curve.png").exists():
            st.image(str(EVAL_DIR / "roc_curve.png"),
                     use_container_width=True)

    with col_ev2:
        st.markdown("#### Calibration curve")
        if (EVAL_DIR / "calibration.png").exists():
            st.image(str(EVAL_DIR / "calibration.png"),
                     use_container_width=True)

        st.markdown("#### Roland Garros backtest (2015–2025)")
        if (EVAL_DIR / "rg_backtest.png").exists():
            st.image(str(EVAL_DIR / "rg_backtest.png"),
                     use_container_width=True)

    st.markdown("---")
    st.markdown("#### Confusion Matrix")
    col_cm1, col_cm2 = st.columns([1, 2])
    with col_cm1:
        if (EVAL_DIR / "confusion_matrix.png").exists():
            st.image(str(EVAL_DIR / "confusion_matrix.png"),
                     use_container_width=True)
    with col_cm2:
        st.markdown("""
        **Reading the confusion matrix:**

        | Cell | Meaning |
        |---|---|
        | True Positive (1342) | Correctly predicted the favourite wins |
        | True Negative (1343) | Correctly predicted the underdog wins |
        | False Positive (457) | Predicted favourite wins, underdog won (missed upset) |
        | False Negative (458) | Predicted underdog wins, favourite won |

        False positives and negatives are almost identical (457 vs 458),
        meaning the model has **no systematic bias** — it treats favourites
        and underdogs fairly.
        """)

    # Roland Garros backtest table
    st.markdown("---")
    st.markdown("#### Roland Garros backtest detail")

    backtest_data = {
        "Year"      : [2015, 2016, 2017, 2018, 2019,
                        2020, 2021, 2022, 2023, 2024, 2025],
        "Predicted" : ["Djokovic", "Djokovic", "Djokovic", "Djokovic",
                        "Djokovic", "Sinner", "Alcaraz", "Alcaraz",
                        "Alcaraz", "Alcaraz", "Alcaraz"],
        "Actual"    : ["Wawrinka", "Djokovic", "Nadal", "Nadal",
                        "Nadal", "Nadal", "Djokovic", "Nadal",
                        "Djokovic", "Alcaraz", "Alcaraz"],
        "Result"    : ["❌ Missed", "✅ Correct", "🟡 Top 3", "🟡 Top 3",
                        "🟡 Top 3", "🟡 Top 3", "🟡 Top 3", "❌ Missed",
                        "🟡 Top 3", "✅ Correct", "✅ Correct"],
        "Actual prob": ["0.1%", "34.3%", "10.7%", "12.4%",
                         "10.5%", "9.7%", "12.6%", "8.4%",
                         "12.7%", "26.2%", "26.8%"],
    }

    st.dataframe(
        pd.DataFrame(backtest_data),
        use_container_width=True,
        hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4: Feature Importance
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.subheader("🔍 Feature Importance")
    st.markdown(
        "Which features does the model rely on most "
        "to predict match outcomes?"
    )

    col_fi1, col_fi2 = st.columns([1, 1])

    with col_fi1:
        if (MODEL_DIR / "feature_importance.png").exists():
            st.image(str(MODEL_DIR / "feature_importance.png"),
                     use_container_width=True)

    with col_fi2:
        st.markdown("""
        #### What each feature means

        | Feature | What it captures |
        |---|---|
        | **diff_clay_elo** | Clay-specific skill built over years — the most reliable signal |
        | **diff_elo** | General tennis skill across all surfaces |
        | **diff_form** | Win rate in last 10 matches — are they peaking right now? |
        | **diff_clay_wr** | Clay win rate over the last 2 years |
        | **diff_h2h_clay** | Head-to-head record on clay specifically |
        | **diff_h2h_overall** | Overall head-to-head record |
        | **diff_rg_best** | Best round ever reached at Roland Garros |
        | **diff_rank** | Current ATP ranking difference |
        | **diff_fatigue** | Days since last match — rest advantage |

        #### Key insight
        **Clay Elo** dominates because it captures long-term
        clay-specific performance. A player who has beaten
        strong clay opponents consistently over years will
        have a high clay Elo — regardless of their general ranking.

        This is why **Sinner** is the top pick — he has the
        highest clay Elo among active players (1900) combined
        with a 100% recent form rate.
        """)

    # Live feature explorer
    st.markdown("---")
    st.markdown("#### 🎛️ Live feature explorer")
    st.markdown(
        "Adjust the sliders to see how each feature affects "
        "win probability in real time."
    )

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("**Player A advantages**")
        clay_elo_diff = st.slider("Clay Elo advantage", -500, 500, 100, 10)
        form_diff     = st.slider("Form advantage",     -1.0,  1.0, 0.2, 0.05)
        clay_wr_diff  = st.slider("Clay WR advantage",  -1.0,  1.0, 0.1, 0.05)

    with col_s2:
        st.markdown("**Other factors**")
        rank_diff     = st.slider("Ranking advantage (lower = better)", -500, 500, 50, 10)
        rg_best_diff  = st.slider("RG history advantage", -7, 7, 1, 1)
        fatigue_diff  = st.slider("Rest days advantage",  -30, 30, 5, 1)

    live_features = {
        "diff_elo"         : clay_elo_diff * 0.5,
        "diff_clay_elo"    : float(clay_elo_diff),
        "diff_form"        : float(form_diff),
        "diff_clay_wr"     : float(clay_wr_diff),
        "diff_h2h_clay"    : 0.0,
        "diff_h2h_overall" : 0.0,
        "diff_rg_best"     : float(rg_best_diff),
        "diff_fatigue"     : float(fatigue_diff),
        "diff_rank"        : float(rank_diff),
    }

    X_live   = pd.DataFrame([live_features])[FEATURE_COLS]
    prob_live = float(model.predict_proba(X_live)[0][1])

    st.markdown("---")
    col_live1, col_live2 = st.columns(2)
    with col_live1:
        st.metric(
            "Player A win probability",
            f"{prob_live:.1%}",
            delta=f"{'favoured' if prob_live > 0.5 else 'underdog'}"
        )
    with col_live2:
        st.metric(
            "Player B win probability",
            f"{1 - prob_live:.1%}",
            delta=f"{'favoured' if prob_live < 0.5 else 'underdog'}"
        )

    # Probability bar
    fig2, ax2 = plt.subplots(figsize=(8, 1.0))
    ax2.barh([0], [prob_live],          color="#1D9E75", height=0.5)
    ax2.barh([0], [1 - prob_live],
             left=[prob_live],          color="#D85A30", height=0.5)
    ax2.set_xlim(0, 1)
    ax2.axis("off")
    ax2.text(prob_live / 2, 0,
             f"Player A  {prob_live:.1%}",
             ha="center", va="center",
             fontsize=10, color="white", fontweight="bold")
    ax2.text(prob_live + (1 - prob_live) / 2, 0,
             f"Player B  {1 - prob_live:.1%}",
             ha="center", va="center",
             fontsize=10, color="white", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()
