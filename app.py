import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------

st.set_page_config(
    page_title="EPL Prediction Dashboard",
    page_icon="⚽",
    layout="wide"
)

# ---------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------

st.sidebar.title("⚽ EPL Analytics")

page = st.sidebar.radio(
    "Go To",
    [
        "Match Prediction",
        "Season Simulation",
        "About Model"
    ]
)

# ---------------------------------------------------
# TEAM LOGOS
# ---------------------------------------------------

team_logos = {

    "Arsenal": "https://upload.wikimedia.org/wikipedia/en/5/53/Arsenal_FC.svg",

     "Aston Villa": "https://cdn.freebiesupply.com/logos/large/2x/aston-villa-logo-png-transparent.png",

    "Bournemouth": "https://upload.wikimedia.org/wikipedia/en/e/e5/AFC_Bournemouth_%282013%29.svg",

    "Brentford": "https://upload.wikimedia.org/wikipedia/en/2/2a/Brentford_FC_crest.svg",

    "Brighton": "https://upload.wikimedia.org/wikipedia/en/f/fd/Brighton_%26_Hove_Albion_logo.svg",

    "Chelsea": "https://upload.wikimedia.org/wikipedia/en/c/cc/Chelsea_FC.svg",

    "Crystal Palace": "https://upload.wikimedia.org/wikipedia/en/a/a2/Crystal_Palace_FC_logo_%282022%29.svg",

    "Everton": "https://upload.wikimedia.org/wikipedia/en/7/7c/Everton_FC_logo.svg",

    "Fulham": "https://upload.wikimedia.org/wikipedia/en/e/eb/Fulham_FC_%28shield%29.svg",

    "Liverpool": "https://upload.wikimedia.org/wikipedia/en/0/0c/Liverpool_FC.svg",

    "Man City": "https://upload.wikimedia.org/wikipedia/en/e/eb/Manchester_City_FC_badge.svg",

    "Man United": "https://upload.wikimedia.org/wikipedia/en/7/7a/Manchester_United_FC_crest.svg",

    "Newcastle": "https://upload.wikimedia.org/wikipedia/en/5/56/Newcastle_United_Logo.svg",

    "Nottm Forest": "https://upload.wikimedia.org/wikipedia/en/e/e5/Nottingham_Forest_F.C._logo.svg",

    "Sunderland": "https://upload.wikimedia.org/wikipedia/en/7/77/Logo_Sunderland.svg",

    "Tottenham": "https://upload.wikimedia.org/wikipedia/en/b/b4/Tottenham_Hotspur.svg",

    "West Ham": "https://upload.wikimedia.org/wikipedia/en/c/c2/West_Ham_United_FC_logo.svg",

    "Wolves": "https://upload.wikimedia.org/wikipedia/en/f/fc/Wolverhampton_Wanderers.svg"
}

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("epl_2025.csv")

    return df

df = load_data()

# ---------------------------------------------------
# TEAM LIST
# ---------------------------------------------------

teams = sorted(df["HomeTeam"].unique())

# ---------------------------------------------------
# ELO CALCULATION
# ---------------------------------------------------

elo_ratings = {}

all_teams = pd.concat([
    df["HomeTeam"],
    df["AwayTeam"]
]).unique()

for team in all_teams:
    elo_ratings[team] = 1500

K = 30

for _, row in df.iterrows():

    home = row["HomeTeam"]
    away = row["AwayTeam"]

    home_goals = row["FTHG"]
    away_goals = row["FTAG"]

    home_elo = elo_ratings[home]
    away_elo = elo_ratings[away]

    expected_home = 1 / (
        1 + 10 ** ((away_elo - home_elo) / 400)
    )

    expected_away = 1 - expected_home

    if home_goals > away_goals:
        actual_home = 1
        actual_away = 0

    elif home_goals < away_goals:
        actual_home = 0
        actual_away = 1

    else:
        actual_home = 0.5
        actual_away = 0.5

    elo_ratings[home] += K * (
        actual_home - expected_home
    )

    elo_ratings[away] += K * (
        actual_away - expected_away
    )

# ---------------------------------------------------
# ATTACK + DEFENSE STRENGTH
# ---------------------------------------------------

avg_home_goals = df["FTHG"].mean()
avg_away_goals = df["FTAG"].mean()

home_attack_strength = (
    df.groupby("HomeTeam")["FTHG"].mean()
    / avg_home_goals
).to_dict()

away_attack_strength = (
    df.groupby("AwayTeam")["FTAG"].mean()
    / avg_away_goals
).to_dict()

home_defense_strength = (
    df.groupby("HomeTeam")["FTAG"].mean()
    / avg_away_goals
).to_dict()

away_defense_strength = (
    df.groupby("AwayTeam")["FTHG"].mean()
    / avg_home_goals
).to_dict()

# ===================================================
# MATCH PREDICTION PAGE
# ===================================================

if page == "Match Prediction":

    st.title("⚽ EPL Prediction Dashboard")

    st.write("AI-powered EPL analytics system")

    st.header("Match Prediction")

    # ---------------------------------------------------
    # TEAM SELECTION
    # ---------------------------------------------------

    home_team = st.selectbox(
        "Select Home Team",
        teams
    )

    away_team = st.selectbox(
        "Select Away Team",
        teams,
        index=1
    )

    # ---------------------------------------------------
    # TEAM LOGOS
    # ---------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            team_logos.get(home_team),
            width=120
        )

        st.write(home_team)

    with col2:

        st.image(
            team_logos.get(away_team),
            width=120
        )

        st.write(away_team)

    # ---------------------------------------------------
    # ELO VALUES
    # ---------------------------------------------------

    home_elo_rating = int(
        elo_ratings.get(home_team, 1500)
    )

    away_elo_rating = int(
        elo_ratings.get(away_team, 1500)
    )

    elo_difference = (
        home_elo_rating - away_elo_rating
    )

    st.subheader("ELO Ratings")

    st.write(
        f"{home_team} ELO: {home_elo_rating}"
    )

    st.write(
        f"{away_team} ELO: {away_elo_rating}"
    )

    st.write(
        f"ELO Difference: {elo_difference}"
    )

    # ---------------------------------------------------
    # ELO COMPARISON CHART
    # ---------------------------------------------------

    elo_fig = px.bar(

        x=[home_team, away_team],

        y=[home_elo_rating, away_elo_rating],

        labels={
            "x": "Team",
            "y": "ELO Rating"
        },

        title="ELO Comparison"
    )

    st.plotly_chart(
        elo_fig,
        use_container_width=True
    )

    # ---------------------------------------------------
    # EXPECTED GOALS
    # ---------------------------------------------------

    home_expected_goals = (

        home_attack_strength[home_team]

        *

        away_defense_strength[away_team]

        *

        avg_home_goals

    )

    away_expected_goals = (

        away_attack_strength[away_team]

        *

        home_defense_strength[home_team]

        *

        avg_away_goals

    )

    st.subheader("Expected Goals (xG)")

    st.write(
        f"{home_team}: {round(home_expected_goals, 2)}"
    )

    st.write(
        f"{away_team}: {round(away_expected_goals, 2)}"
    )

    # ---------------------------------------------------
    # XG CHART
    # ---------------------------------------------------

    xg_fig = px.bar(

        x=[home_team, away_team],

        y=[home_expected_goals, away_expected_goals],

        labels={
            "x": "Team",
            "y": "Expected Goals"
        },

        title="Expected Goals Comparison"
    )

    st.plotly_chart(
        xg_fig,
        use_container_width=True
    )

    # ---------------------------------------------------
    # POISSON MODEL
    # ---------------------------------------------------

    max_goals = 6

    home_goal_probs = [
        poisson.pmf(i, home_expected_goals)
        for i in range(max_goals + 1)
    ]

    away_goal_probs = [
        poisson.pmf(i, away_expected_goals)
        for i in range(max_goals + 1)
    ]

    home_win_prob = 0
    draw_prob = 0
    away_win_prob = 0

    for i in range(max_goals + 1):

        for j in range(max_goals + 1):

            probability = (
                home_goal_probs[i]
                *
                away_goal_probs[j]
            )

            if i > j:
                home_win_prob += probability

            elif i == j:
                draw_prob += probability

            else:
                away_win_prob += probability

    # ---------------------------------------------------
    # RESULTS
    # ---------------------------------------------------

    st.subheader("Prediction Results")

    st.write(
        f"{home_team} Win Probability: "
        f"{round(home_win_prob * 100, 2)}%"
    )

    st.write(
        f"Draw Probability: "
        f"{round(draw_prob * 100, 2)}%"
    )

    st.write(
        f"{away_team} Win Probability: "
        f"{round(away_win_prob * 100, 2)}%"
    )

    # ---------------------------------------------------
    # CONFIDENCE GAUGE
    # ---------------------------------------------------

    confidence = max(
        home_win_prob,
        draw_prob,
        away_win_prob
    ) * 100

    gauge_fig = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=confidence,

            title={
                "text": "Prediction Confidence"
            },

            gauge={
                "axis": {
                    "range": [0, 100]
                },

                "bar": {
                    "color": "green"
                }
            }
        )
    )

    st.plotly_chart(
        gauge_fig,
        use_container_width=True
    )

    # ---------------------------------------------------
    # RESULT CHART
    # ---------------------------------------------------

    result_fig = px.bar(

        x=[
            f"{home_team} Win",
            "Draw",
            f"{away_team} Win"
        ],

        y=[
            home_win_prob * 100,
            draw_prob * 100,
            away_win_prob * 100
        ],

        labels={
            "x": "Result",
            "y": "Probability (%)"
        },

        title="Match Outcome Probabilities"
    )

    st.plotly_chart(
        result_fig,
        use_container_width=True
    )

# ===================================================
# SEASON SIMULATION PAGE
# ===================================================

elif page == "Season Simulation":

    st.title("🏆 Full EPL Season Simulation")

    if st.button("Simulate Entire EPL Season"):

        with st.spinner(
            "Running Monte Carlo Simulations..."
        ):

            simulation_results = []

            for sim in range(200):

                table = {
                    team: 0
                    for team in teams
                }

                for _, row in df.iterrows():

                    home_team_sim = row["HomeTeam"]
                    away_team_sim = row["AwayTeam"]

                    home_lambda = (

                        home_attack_strength[home_team_sim]

                        *

                        away_defense_strength[away_team_sim]

                        *

                        avg_home_goals
                    )

                    away_lambda = (

                        away_attack_strength[away_team_sim]

                        *

                        home_defense_strength[home_team_sim]

                        *

                        avg_away_goals
                    )

                    home_goals = np.random.poisson(
                        home_lambda
                    )

                    away_goals = np.random.poisson(
                        away_lambda
                    )

                    if home_goals > away_goals:

                        table[home_team_sim] += 3

                    elif home_goals < away_goals:

                        table[away_team_sim] += 3

                    else:

                        table[home_team_sim] += 1
                        table[away_team_sim] += 1

                sorted_table = sorted(
                    table.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                simulation_results.append(
                    sorted_table
                )

            # ---------------------------------------------------
            # TITLE PROBABILITIES
            # ---------------------------------------------------

            title_counts = {
                team: 0
                for team in teams
            }

            for sim in simulation_results:

                winner = sim[0][0]

                title_counts[winner] += 1

            title_df = pd.DataFrame({

                "Team": list(title_counts.keys()),

                "Title Probability (%)": [

                    round(
                        count / len(simulation_results)
                        * 100,
                        2
                    )

                    for count in title_counts.values()
                ]
            })

            title_df = title_df.sort_values(
                by="Title Probability (%)",
                ascending=False
            )

            st.subheader("🏆 Title Probabilities")

            st.dataframe(title_df)

            # ---------------------------------------------------
            # TOP 4 PROBABILITY
            # ---------------------------------------------------

            top4_counts = {
                team: 0
                for team in teams
            }

            for sim in simulation_results:

                top4 = [
                    x[0]
                    for x in sim[:4]
                ]

                for team in top4:
                    top4_counts[team] += 1

            top4_df = pd.DataFrame({

                "Team": list(top4_counts.keys()),

                "Top 4 Probability (%)": [

                    round(
                        count / len(simulation_results)
                        * 100,
                        2
                    )

                    for count in top4_counts.values()
                ]
            })

            top4_df = top4_df.sort_values(
                by="Top 4 Probability (%)",
                ascending=False
            )

            st.subheader("⭐ Top 4 Probabilities")

            st.dataframe(top4_df)

            # ---------------------------------------------------
            # RELEGATION
            # ---------------------------------------------------

            relegation_counts = {
                team: 0
                for team in teams
            }

            for sim in simulation_results:

                relegated = [
                    x[0]
                    for x in sim[-3:]
                ]

                for team in relegated:
                    relegation_counts[team] += 1

            relegation_df = pd.DataFrame({

                "Team": list(relegation_counts.keys()),

                "Relegation Probability (%)": [

                    round(
                        count / len(simulation_results)
                        * 100,
                        2
                    )

                    for count in relegation_counts.values()
                ]
            })

            relegation_df = relegation_df.sort_values(
                by="Relegation Probability (%)",
                ascending=False
            )

            st.subheader("📉 Relegation Probabilities")

            st.dataframe(relegation_df)

            # ---------------------------------------------------
            # FINAL TABLE
            # ---------------------------------------------------

            avg_points = {
                team: 0
                for team in teams
            }

            for sim in simulation_results:

                for team, points in sim:

                    avg_points[team] += points

            for team in avg_points:

                avg_points[team] /= len(
                    simulation_results
                )

            final_table = pd.DataFrame({

                "Team": list(avg_points.keys()),

                "Predicted Points": [

                    round(points, 1)

                    for points in avg_points.values()
                ]
            })

            final_table = final_table.sort_values(
                by="Predicted Points",
                ascending=False
            )

            st.subheader("📊 Predicted Final EPL Table")

            st.dataframe(final_table)

            # ---------------------------------------------------
            # TITLE CHART
            # ---------------------------------------------------

            title_chart = px.bar(

                title_df.head(10),

                x="Team",

                y="Title Probability (%)",

                title="EPL Title Chances"
            )

            st.plotly_chart(
                title_chart,
                use_container_width=True
            )

            # ---------------------------------------------------
            # TOP 4 CHART
            # ---------------------------------------------------

            top4_chart = px.bar(

                top4_df.head(10),

                x="Team",

                y="Top 4 Probability (%)",

                title="Top 4 Chances"
            )

            st.plotly_chart(
                top4_chart,
                use_container_width=True
            )

            # ---------------------------------------------------
            # RELEGATION CHART
            # ---------------------------------------------------

            relegation_chart = px.bar(

                relegation_df.head(10),

                x="Team",

                y="Relegation Probability (%)",

                title="Relegation Chances"
            )

            st.plotly_chart(
                relegation_chart,
                use_container_width=True
            )

# ===================================================
# ABOUT PAGE
# ===================================================

elif page == "About Model":

    st.title("📘 About This Model")

    st.write("""
    This EPL Prediction Dashboard uses:

    - ELO Rating System
    - Poisson Goal Model
    - Expected Goals (xG)
    - Monte Carlo Season Simulation
    - Match Probability Analysis
    - Team Strength Modeling

    Built using:
    - Python
    - Streamlit
    - Plotly
    - Pandas
    - NumPy
    """)

    st.subheader("👨‍💻 Developer")

    st.write("JP Steve Akash")
    st.write("AI and Data Science enthusiast")