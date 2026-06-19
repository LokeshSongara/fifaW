import pandas as pd
import streamlit as st
import joblib
import random

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="FIFA 2026 Predictor",
    layout="wide"
)

# =========================
# LOAD MODEL + DATA
# =========================
model = joblib.load("models/fifa_rf_model.pkl")

df = pd.read_csv("data/results.csv")
df["date"] = pd.to_datetime(df["date"])

team_stats = pd.read_csv("data/team_stats.csv", index_col=0)
elo_df = pd.read_csv("data/fifa_ranking.csv")

elo_dict = dict(zip(elo_df["team"], elo_df["points"]))
team_stats_dict = team_stats.to_dict(orient="index")

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.title("⚙ Controls")

selected_date = st.sidebar.date_input("Select Match Date")

mode = st.sidebar.radio(
    "Mode",
    ["📊 Match Predictor", "🏆 World Cup Simulator"]
)

matches = df[df["date"].dt.date == selected_date]

# =========================
# HEADER
# =========================
st.title("⚽ FIFA World Cup 2026 AI Predictor")
st.markdown("AI-powered match prediction + tournament simulation system")

# =========================
# MODEL FUNCTION
# =========================
def predict_match(home, away):

    home_stats = team_stats_dict[home]
    away_stats = team_stats_dict[away]

    features = pd.DataFrame([{
        "home_form": 0.5,
        "away_form": 0.5,
        "home_goal_diff": home_stats["goal_difference"],
        "away_goal_diff": away_stats["goal_difference"],
        "home_win_rate": home_stats["win_rate"],
        "away_win_rate": away_stats["win_rate"],
        "neutral": 0
    }])

    features = features[model.feature_names_in_]

    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0]

    return pred, prob

# =========================
# MATCH PREDICTOR MODE
# =========================
if mode == "📊 Match Predictor":

    st.subheader("📅 Matches on Selected Date")

    if len(matches) == 0:
        st.warning("No matches found for selected date")
    else:

        for _, row in matches.iterrows():

            col1, col2, col3 = st.columns([3, 2, 3])

            with col1:
                st.markdown(f"### {row['home_team']}")

            with col2:
                st.markdown("## 🆚")

            with col3:
                st.markdown(f"### {row['away_team']}")

            pred, prob = predict_match(row["home_team"], row["away_team"])

            st.success(f"🏆 Prediction: **{pred}**")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric("Home Win", f"{prob[2]:.2%}")
                st.progress(float(prob[2]))

            with c2:
                st.metric("Draw", f"{prob[1]:.2%}")
                st.progress(float(prob[1]))

            with c3:
                st.metric("Away Win", f"{prob[0]:.2%}")
                st.progress(float(prob[0]))

            st.divider()

# =========================
# SIMULATION MODE
# =========================
def simulate_match(home, away):

    pred, prob = predict_match(home, away)

    r = random.random()

    if r < prob[2]:
        return home
    elif r < prob[2] + prob[1]:
        return "Draw"
    else:
        return away


if mode == "🏆 World Cup Simulator":

    st.subheader("🏆 FIFA World Cup Simulation Engine")

    teams = [
        "Brazil", "Argentina", "France", "Germany",
        "Spain", "England", "Portugal", "Netherlands"
    ]

    st.info("Click button to simulate full mini tournament")

    if st.button("🚀 Run Simulation"):

        winners = []

        for i in range(0, len(teams), 2):

            home = teams[i]
            away = teams[i + 1]

            winner = simulate_match(home, away)

            col1, col2, col3 = st.columns([3, 1, 3])

            with col1:
                st.write(home)

            with col2:
                st.write("🆚")

            with col3:
                st.write(away)

            st.success(f"🏆 Winner: {winner}")

            winners.append(winner)

        st.markdown("### 🏁 Simulation Result Summary")
        st.write(winners)


        import os
import streamlit as st

st.write(os.listdir("."))
st.write(os.listdir("models"))
import os
import joblib

BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "models", "fifa_rf_model.pkl")

model = joblib.load(model_path)