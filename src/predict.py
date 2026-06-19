import pandas as pd
import joblib
import os

results_df = pd.read_csv("data/results.csv")

results_df["date"] = pd.to_datetime(results_df["date"])

results_df = results_df.sort_values("date")


def calculate_form(results):

    if len(results) == 0:
        return 0.5

    points = 0

    for result in results[-5:]:

        if result == "W":
            points += 3

        elif result == "D":
            points += 1

    return points / 15


def get_recent_form(team):

    team_matches = results_df[
        (results_df["home_team"] == team)
        |
        (results_df["away_team"] == team)
    ]

    team_matches = team_matches.tail(5)

    results = []

    for _, match in team_matches.iterrows():

        if match["home_team"] == team:

            if match["home_score"] > match["away_score"]:
                results.append("W")

            elif match["home_score"] < match["away_score"]:
                results.append("L")

            else:
                results.append("D")

        else:

            if match["away_score"] > match["home_score"]:
                results.append("W")

            elif match["away_score"] < match["home_score"]:
                results.append("L")

            else:
                results.append("D")

    return calculate_form(results)


print("Brazil:", get_recent_form("Brazil"))
print("Argentina:", get_recent_form("Argentina"))
model = joblib.load(
    "models/fifa_rf_model.pkl"
)

team_stats = pd.read_csv(
    "data/team_stats.csv",
    index_col=0
)
def predict_match(home_team, away_team, neutral=0):

    home_stats = team_stats.loc[home_team]
    away_stats = team_stats.loc[away_team]

    home_form = get_recent_form(home_team)
    away_form = get_recent_form(away_team)

    print(f"\n{home_team} Form: {home_form:.2f}")
    print(f"{away_team} Form: {away_form:.2f}")

    features = [[
        home_stats["win_rate"],
        away_stats["win_rate"],
        home_stats["goal_difference"],
        away_stats["goal_difference"],
        home_form,
        away_form,
        neutral
    ]]

    prediction = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]

    print("\nProbabilities:")

    for label, prob in zip(
        model.classes_,
        probabilities
    ):
        print(f"{label}: {prob:.2%}")

    return prediction
result = predict_match(
    "Brazil",
    "Argentina"
)

print("\nPrediction:", result)