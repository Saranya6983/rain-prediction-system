import streamlit as st
import pickle
import numpy as np
import plotly.express as px
import pandas as pd
import requests

# Load trained model
model = pickle.load(open("rain_model.pkl", "rb"))

# Load dataset
data = pd.read_csv("weather.csv")

# OpenWeather API Key
API_KEY = "bd276c7f73ba6838d08eb291b4919b1a"

# Page settings
st.set_page_config(
    page_title="Rain Prediction App",
    page_icon="🌧️",
    layout="centered"
)

# Sidebar
st.sidebar.title("🌦️ Dashboard Menu")

st.sidebar.info(
    """
    Rain Prediction System
    
    Features:
    - Live Weather
    - AI Rain Prediction
    - Forecast Trends
    - Weather Visualization
    """
)

# Title
st.title("🌧️ Rain Prediction System")

st.markdown("### Enter Weather Details")

# City input
city = st.text_input(
    "Enter City Name",
    "Chennai"
)

# Live weather button
if st.button("Get Live Weather"):

    # Current Weather API
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    response = requests.get(url)

    weather_data = response.json()

    if response.status_code == 200:

        temp = weather_data['main']['temp']
        humidity_live = weather_data['main']['humidity']
        pressure_live = weather_data['main']['pressure']
        wind_live = weather_data['wind']['speed']

        st.success(f"✅ Live Weather in {city}")

        # Metric Cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "🌡 Temp",
                f"{temp} °C"
            )

        with col2:
            st.metric(
                "💧 Humidity",
                f"{humidity_live}%"
            )

        with col3:
            st.metric(
                "📈 Pressure",
                f"{pressure_live} hPa"
            )

        with col4:
            st.metric(
                "💨 Wind",
                f"{wind_live} m/s"
            )

        st.divider()

        # Live AI Prediction
        live_features = np.array([[
            temp - 5,
            temp + 5,
            humidity_live,
            pressure_live,
            wind_live
        ]])

        live_prediction = model.predict(
            live_features
        )

        live_probability = model.predict_proba(
            live_features
        )

        rain_chance = (
            live_probability[0][1] * 100
        )

        st.subheader("🤖 Live AI Prediction")

        if live_prediction[0] == 1:
            st.success("🌧️ Rain Expected")
        else:
            st.warning("☀️ No Rain Expected")

        st.info(
            f"Rain Probability: {rain_chance:.2f}%"
        )

        # Forecast API
        forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"

        forecast_response = requests.get(
            forecast_url
        )

        forecast_data = forecast_response.json()

        if forecast_response.status_code == 200:

            forecast_list = forecast_data['list']

            dates = []
            temperatures = []

            for item in forecast_list[:10]:

                dates.append(
                    item['dt_txt']
                )

                temperatures.append(
                    item['main']['temp']
                )

            forecast_df = pd.DataFrame({
                "Date": dates,
                "Temperature": temperatures
            })

            st.subheader(
                "📅 Forecast Temperature Trend"
            )

            fig_forecast = px.line(
                forecast_df,
                x="Date",
                y="Temperature",
                title="Upcoming Temperature Forecast",
                markers=True
            )

            st.plotly_chart(
                fig_forecast,
                use_container_width=True
            )

    else:
        st.error(weather_data["message"])

# Divider
st.divider()

# Manual Prediction Section
st.subheader("🛠 Manual Weather Prediction")

min_temp = st.slider(
    "Min Temperature",
    -10.0,
    40.0,
    20.0
)

max_temp = st.slider(
    "Max Temperature",
    0.0,
    50.0,
    30.0
)

humidity = st.slider(
    "Humidity 9AM",
    0.0,
    100.0,
    85.0
)

pressure = st.slider(
    "Pressure 9AM",
    980.0,
    1050.0,
    1005.0
)

wind = st.slider(
    "Wind Speed 9AM",
    0.0,
    100.0,
    15.0
)

# Manual Prediction Button
if st.button("Predict Rain"):

    features = np.array([[
        min_temp,
        max_temp,
        humidity,
        pressure,
        wind
    ]])

    prediction = model.predict(features)

    probability = model.predict_proba(features)

    rain_probability = (
        probability[0][1] * 100
    )

    st.subheader("📌 Prediction Result")

    if prediction[0] == 1:
        st.success(
            "🌧️ Rain Expected Tomorrow"
        )
    else:
        st.warning(
            "☀️ No Rain Expected"
        )

    st.info(
        f"Rain Probability: {rain_probability:.2f}%"
    )

# Divider
st.divider()

# Visualization Section
st.markdown(
    "## 📊 Weather Data Visualization"
)

# Humidity Distribution
fig1 = px.histogram(
    data,
    x="Humidity9am",
    title="Humidity Distribution",
    height=400
)

st.plotly_chart(
    fig1,
    use_container_width=True
)

# Temperature vs Rain
fig2 = px.scatter(
    data,
    x="MinTemp",
    y="MaxTemp",
    color="RainTomorrow",
    title="Temperature vs Rain",
    height=500
)

st.plotly_chart(
    fig2,
    use_container_width=True
)
