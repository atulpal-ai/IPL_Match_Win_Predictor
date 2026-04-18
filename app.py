import streamlit as st
import pandas as pd
import pickle

# model load
pipe = pickle.load(open('pipe.pkl', 'rb'))

st.title("🏏 IPL Win Predictor")

# dropdown inputs
batting_team = st.selectbox("Batting Team", [
    'Chennai Super Kings','Delhi Capitals','Kings XI Punjab',
    'Kolkata Knight Riders','Mumbai Indians','Rajasthan Royals',
    'Royal Challengers Bangalore','Sunrisers Hyderabad'
])

bowling_team = st.selectbox("Bowling Team", [
    'Chennai Super Kings','Delhi Capitals','Kings XI Punjab',
    'Kolkata Knight Riders','Mumbai Indians','Rajasthan Royals',
    'Royal Challengers Bangalore','Sunrisers Hyderabad'
])

city = st.selectbox("City", [
    'Mumbai','Delhi','Kolkata','Chennai','Bangalore','Hyderabad'
])

# numeric inputs
runs_left = st.number_input("Runs Left")
balls_left = st.number_input("Balls Left")
wickets = st.number_input("Wickets Left")
total_runs = st.number_input("Target")

# calculate rates
crr = (total_runs - runs_left) / (120 - balls_left) * 6 if balls_left != 120 else 0
rrr = (runs_left * 6) / balls_left if balls_left != 0 else 0

# predict button
if st.button("Predict Probability"):
    input_df = pd.DataFrame({
        'batting_team':[batting_team],
        'bowling_team':[bowling_team],
        'city':[city],
        'runs_left':[runs_left],
        'balls_left':[balls_left],
        'wickets':[wickets],
        'total_runs_x':[total_runs],
        'crr':[crr],
        'rrr':[rrr]
    })

    result = pipe.predict_proba(input_df)

    st.header(f"Win Probability: {round(result[0][1]*100,2)}%")
