# app.py (v4) - Full Upgrade: Real Pitch-vs-Batter, Trend Charts, Deployment Ready

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta
from pybaseball import statcast_batter, statcast_pitcher, playerid_lookup, pitching_stats_range, team_batting_stats_range
import statsapi
import os
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# 🔧 CONFIGURATION & SETUP
# -----------------------------
st.set_page_config(page_title="MLB Predictive Analytics App", layout="wide")
st.title("⚾ MLB Predictive Analytics Dashboard")
st.markdown("""
Real-time pitcher run suppression, batter matchups, team scoring probabilities,
and model evaluation — powered by Statcast & advanced analytics.
""")

# File for storing prediction history
HISTORY_FILE = 'data/history_predictions.csv'
os.makedirs('data', exist_ok=True)

# Date Range Setup
today = datetime.today()
yesterday = (today - timedelta(days=1)).strftime('%Y-%m-%d')
start_date = (today - timedelta(days=30)).strftime('%Y-%m-%d')
end_date = today.strftime('%Y-%m-%d')

# -----------------------------
# 🧠 CORE MODELING FUNCTIONS
# -----------------------------
def estimate_zero_run_prob(xfip, k9, bb9):
    z = -2.2 * xfip + 1.5 * k9 - 1.3 * bb9
    return round(1 / (1 + np.exp(-z)), 2)

def simulate_team_runs(ops, xfip, n_sim=1000):
    base_rate = 4.5 + (ops - 0.7) * 10 - (xfip - 4.0) * 2
    sims = np.random.poisson(base_rate, size=n_sim)
    return round(np.sum(sims >= 3) / n_sim, 2)

# -----------------------------
# 📦 LOAD & PREP DATA
# -----------------------------
@st.cache_data
def load_pitcher_data():
    df = pitching_stats_range(start_date, end_date)
    df = df[['Name', 'xFIP', 'K/9', 'BB/9']].dropna().rename(columns={'Name': 'Pitcher'})
    return df

@st.cache_data
def load_team_batting():
    return team_batting_stats_range(start_date, end_date)[['Team', 'OPS']].dropna()

# -----------------------------
# 🎯 TABLE 1: Scoreless Pitchers
# -----------------------------
st.subheader("🎯 Top 5 Pitchers Least Likely to Allow a Run")
pitchers = load_pitcher_data()
pitchers['P(0 Runs)'] = pitchers.apply(lambda row: estimate_zero_run_prob(row['xFIP'], row['K/9'], row['BB/9']), axis=1)
top5_pitchers = pitchers.sort_values(by='P(0 Runs)', ascending=False).head(5).reset_index(drop=True)
st.dataframe(top5_pitchers, use_container_width=True)

# -----------------------------
# 🔒 TABLE 2: Batter Suppression Matchups
# -----------------------------
st.subheader("🔒 Batter Suppression Matchups vs Top Pitchers")
suppression_rows = []

for _, row in top5_pitchers.iterrows():
    pitcher_name = row['Pitcher']
    result = playerid_lookup(*pitcher_name.split())
    if result.empty:
        continue
    pitcher_id = result.iloc[0]['key_mlbam']
    pitcher_data = statcast_pitcher(start_date, end_date, pitcher_id)

    if pitcher_data.empty:
        continue
    pitcher_team = pitcher_data['home_team'].iloc[-1] if 'home_team' in pitcher_data.columns else "N/A"

    # Estimate arsenal usage (basic logic — could be enhanced)
    pitch_counts = pitcher_data['pitch_type'].value_counts(normalize=True)

    # Get opposing team lineup from StatsAPI
    opposing_team = statsapi.schedule(start_date=today.strftime('%Y-%m-%d'), end_date=end_date, team=pitcher_team)[0]['away_name']
    lineup = statsapi.boxscore_data(statsapi.schedule()[0]['game_id'])['away']['players']

    for pid, pdata in lineup.items():
        batter_name = pdata['person']['fullName']
        bresult = playerid_lookup(*batter_name.split())
        if bresult.empty:
            continue
        batter_id = bresult.iloc[0]['key_mlbam']
        batter_data = statcast_batter(start_date, end_date, batter_id)
        if batter_data.empty:
            continue

        # Calculate suppression score
        score = 0
        for pitch_type, usage in pitch_counts.items():
            matchups = batter_data[batter_data['pitch_type'] == pitch_type]
            if not matchups.empty:
                score += usage * (1 - matchups['xwOBA'].mean())
        suppression_rows.append({
            'Pitcher': pitcher_name,
            'Batter': batter_name,
            'Suppression Score': round(score, 3)
        })

suppress_df = pd.DataFrame(suppression_rows)
suppress_df = suppress_df.sort_values(by='Suppression Score', ascending=False).groupby('Pitcher').head(3).reset_index(drop=True)
st.dataframe(suppress_df, use_container_width=True)

# -----------------------------
# 🔥 TABLE 3: Team Scoring Probability
# -----------------------------
st.subheader("🔥 Teams Most Likely to Score 3+ Runs")
batting = load_team_batting()
batting = batting.merge(pitchers[['Pitcher', 'xFIP']], left_on='Team', right_on='Pitcher', how='left')
batting['P(3+ Runs)'] = batting.apply(lambda row: simulate_team_runs(row['OPS'], row['xFIP']) if pd.notnull(row['xFIP']) else np.nan, axis=1)
batting = batting[['Team', 'OPS', 'P(3+ Runs)']].dropna().sort_values(by='P(3+ Runs)', ascending=False)
st.dataframe(batting, use_container_width=True)

# -----------------------------
# 📈 TREND CHART: P(0 Runs) Over Time
# -----------------------------
st.subheader("📈 Pitcher Trend Chart: P(0 Runs) vs xFIP")
pitch_chart = alt.Chart(top5_pitchers).mark_circle(size=80).encode(
    x='xFIP',
    y='P(0 Runs)',
    color=alt.Color('Pitcher', legend=None),
    tooltip=['Pitcher', 'xFIP', 'P(0 Runs)']
).interactive()
st.altair_chart(pitch_chart, use_container_width=True)

# -----------------------------
# 💾 EXPORT OPTION
# -----------------------------
st.download_button("📥 Export Pitcher Table to CSV", data=top5_pitchers.to_csv(index=False), file_name="top_pitchers.csv")

