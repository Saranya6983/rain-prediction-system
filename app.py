import streamlit as st
import pickle
import numpy as np
import plotly.express as px
import pandas as pd
import requests
from streamlit_js_eval import streamlit_js_eval

# PAGE SETTINGS
st.set_page_config(
    page_title="Rain Prediction App",
    page_icon="🌧️",
    layout="wide"
)

# LOAD MODEL
model = pickle.load(open("rain_model.pkl", "rb"))

# LOAD DATASET
data = pd.read_csv("weatherAUS.csv")

# API KEY
API_KEY = "bd276c7f73ba6838d08eb291b4919b1a"

# SIDEBAR
st.sidebar.title("🌦️ Dashboard Menu")

st.sidebar.info("""
Rain Prediction System

Features:
• Live Weather
• AI Rain Prediction
• Forecast Trends
• Weather Visualization
""")

# TITLE
st.title("🌧️ Rain Prediction System")

st.markdown("## Enter Weather Details")

# CITY INPUT
city = st.text_input(
    "Enter City / Area",
    "Chennai"
)

st.caption(
    "🌍 Supports live weather prediction for cities worldwide"
)

# GPS LOCATION
lat = None
lon = None

try:

    location = streamlit_js_eval(
        js_expressions="""
        navigator.geolocation.getCurrentPosition(
            (pos) => {
                return {
                    latitude: pos.coords.latitude,
                    longitude: pos.coords.longitude
                }
            }
        )
        """,
        key="get_location"
    )

    if location:

        lat = location["latitude"]
        lon = location["longitude"]

        st.success("✅ Live GPS Location Detected")

        st.write(f"Latitude: {lat}")
        st.write(f"Longitude: {lon}")

except:
    pass

# LIVE WEATHER
if st.button("Get Live Weather"):

    # GPS MODE
    if lat and lon:

        url = (
            f"https://api.openweathermap.org/data/2.5/weather?"
            f"lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        )

        forecast_url = (
            f"https://api.openweathermap.org/data/2.5/forecast?"
            f"lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        )

    # CITY MODE
    else:

        url = (
            f"https://api.openweathermap.org/data/2.5/weather?"
            f"q={city}&appid={API_KEY}&units=metric"
        )

        forecast_url = (
            f"https://api.openweathermap.org/data/2.5/forecast?"
            f"q={city}&appid={API_KEY}&units=metric"
        )

    # CURRENT WEATHER
    response = requests.get(url)

    weather_data = response.json()

    if response.status_code == 200:

        city_name = weather_data["name"]

        temp = weather_data['main']['temp']
        humidity_live = weather_data['main']['humidity']
        pressure_live = weather_data['main']['pressure']
        wind_live = weather_data['wind']['speed']

        st.success(f"✅ Live Weather in {city_name}")

        # METRICS
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "🌡 Temperature",
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

        # AI PREDICTION
        live_features = np.array([[
            temp - 5,
            temp + 5,
            humidity_live,
            pressure_live,
            wind_live
        ]])

        prediction = model.predict(
            live_features
        )

        probability = model.predict_proba(
            live_features
        )

        rain_probability = probability[0][1] * 100

        st.subheader("🤖 Live AI Prediction")

        if prediction[0] == 1:
            st.success("🌧️ Rain Expected")
        else:
            st.warning("☀️ No Rain Expected")

        st.info(
            f"Rain Probability: {rain_probability:.2f}%"
        )

        # FORECAST API
        forecast_response = requests.get(
            forecast_url
        )

        forecast_data = forecast_response.json()

        if forecast_response.status_code == 200:

            forecast_list = forecast_data['list']

            dates = []
            temperatures = []
            humidities = []
            pressures = []

            for item in forecast_list[:10]:

                dates.append(item['dt_txt'])

                temperatures.append(
                    item['main']['temp']
                )

                humidities.append(
                    item['main']['humidity']
                )

                pressures.append(
                    item['main']['pressure']
                )

            forecast_df = pd.DataFrame({
                "Date": dates,
                "Temperature": temperatures,
                "Humidity": humidities,
                "Pressure": pressures
            })

            st.divider()

            # TEMPERATURE TREND
            st.subheader(
                "📅 Forecast Temperature Trend"
            )

            fig_temp = px.line(
                forecast_df,
                x="Date",
                y="Temperature",
                title="Upcoming Temperature Forecast",
                markers=True
            )

            st.plotly_chart(
                fig_temp,
                use_container_width=True
            )

            # HUMIDITY TREND
            st.subheader(
                "💧 Humidity Trend"
            )

            fig_humidity = px.line(
                forecast_df,
                x="Date",
                y="Humidity",
                title="Humidity Forecast",
                markers=True
            )

            st.plotly_chart(
                fig_humidity,
                use_container_width=True
            )

            # PRESSURE TREND
            st.subheader(
                "📈 Pressure Trend"
            )

            fig_pressure = px.line(
                forecast_df,
                x="Date",
                y="Pressure",
                title="Pressure Forecast",
                markers=True
            )

            st.plotly_chart(
                fig_pressure,
                use_container_width=True
            )

    else:
        st.error("Unable to fetch weather data")

# DIVIDER
st.divider()

# MANUAL PREDICTION
st.subheader("🛠️ Manual Weather Prediction")

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

# MANUAL BUTTON
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

    rain_probability = probability[0][1] * 100

    st.subheader("📌 Prediction Result")

    if prediction[0] == 1:
        st.success("🌧️ Rain Expected Tomorrow")
    else:
        st.warning("☀️ No Rain Expected")

    st.info(
        f"Rain Probability: {rain_probability:.2f}%"
    )

# VISUALIZATION
st.divider()

st.markdown("## 📊 Weather Data Visualization")

# HUMIDITY GRAPH
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

# TEMPERATURE GRAPH
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
