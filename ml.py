"""
Output layer:
Streamlit dashboard for cold-chain visualization and anomaly detection.
"""

import sys
import time
import requests
import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, "src")

import config
import ml


@st.cache_data(ttl=10)
def get_data(hours: int) -> pd.DataFrame:
    cfg = config.load()

    influx_url = cfg["influxdb"]["url"].rstrip("/")
    database = cfg["influxdb"]["database"]
    token = cfg["influxdb"].get("token", "")

    query = f"""
    SELECT
        time,
        vehicle_id,
        zone,
        alert,
        fault,
        temperature,
        humidity
    FROM temperature_reading
    WHERE time >= now() - INTERVAL '{hours} hours'
    ORDER BY time DESC
    LIMIT 10000
    """

    url = f"{influx_url}/api/v3/query_sql"

    headers = {
        "Content-Type": "application/json"
    }

    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        response = requests.post(
            url,
            headers=headers,
            json={
                "db": database,
                "q": query
            },
            timeout=10
        )

        if response.status_code != 200:
            st.error(
                f"InfluxDB query failed: {response.status_code} - {response.text}"
            )
            return pd.DataFrame()

        data = response.json()

    except Exception as error:
        st.error(f"InfluxDB query failed: {error}")
        return pd.DataFrame()

    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, dict) and "data" in data:
        df = pd.DataFrame(data["data"])
    else:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    df["humidity"] = pd.to_numeric(df["humidity"], errors="coerce")

    required_columns = [
        "time",
        "vehicle_id",
        "zone",
        "alert",
        "fault",
        "temperature",
        "humidity"
    ]

    available_columns = [
        column for column in required_columns
        if column in df.columns
    ]

    return (
        df[available_columns]
        .dropna(subset=["time", "temperature", "humidity"])
        .sort_values("time")
        .reset_index(drop=True)
    )


def main():
    st.set_page_config(
        page_title="Cold Chain Monitor",
        layout="wide"
    )

    st.title("Cold Chain Temperature Monitor")

    st.write(
        "Output layer: Streamlit dashboard showing real-time cold-chain data, "
        "temperature trends, zone analysis, and machine learning anomaly detection."
    )

    hours = st.sidebar.slider(
        "Data window in hours",
        min_value=1,
        max_value=24,
        value=1
    )

    auto_refresh = st.sidebar.checkbox(
        "Auto-refresh every 10 seconds",
        value=True
    )

    df = get_data(hours)

    if df.empty:
        st.warning(
            "No data found. Make sure InfluxDB, Mosquitto, subscriber.py, "
            "and simulator.py are running."
        )
        return

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Vehicles", df["vehicle_id"].nunique())
    col3.metric("Zones", df["zone"].nunique())

    if "alert" in df.columns:
        alert_count = df[
            df["alert"].astype(str).str.lower() == "true"
        ].shape[0]
    else:
        alert_count = 0

    col4.metric("Cold Chain Alerts", alert_count)

    st.divider()

    st.subheader("Visualization 1: Temperature Over Time by Vehicle")

    temperature_pivot = (
        df.set_index("time")
        .groupby([pd.Grouper(freq="1min"), "vehicle_id"])["temperature"]
        .mean()
        .unstack("vehicle_id")
    )

    st.line_chart(temperature_pivot)

    st.divider()

    st.subheader("Visualization 2: Temperature Distribution by Zone")

    zone_stats = (
        df.groupby("zone")["temperature"]
        .agg(["mean", "min", "max", "std", "count"])
        .round(2)
        .rename(columns={
            "mean": "Mean °C",
            "min": "Min °C",
            "max": "Max °C",
            "std": "Std °C",
            "count": "Records"
        })
    )

    st.dataframe(zone_stats, use_container_width=True)
    st.bar_chart(zone_stats["Mean °C"])

    st.divider()

    st.subheader("Machine Learning: Isolation Forest Anomaly Detection")

    st.caption(
        "The ML model uses temperature and humidity to identify abnormal sensor readings."
    )

    df_ml = ml.detect_anomalies(df)

    if "is_anomaly" not in df_ml.columns:
        st.info("At least 50 records are needed to run anomaly detection.")
    else:
        anomaly_count = int(df_ml["is_anomaly"].sum())

        metric1, metric2 = st.columns(2)

        metric1.metric("Anomalies Detected", anomaly_count)
        metric2.metric("Anomaly Rate", f"{anomaly_count / len(df_ml):.1%}")

        chart = (
            alt.Chart(df_ml)
            .mark_point(opacity=0.7, size=45)
            .encode(
                x=alt.X("temperature:Q", title="Temperature (°C)"),
                y=alt.Y("humidity:Q", title="Humidity (%)"),
                color=alt.Color("is_anomaly:N", title="Anomaly"),
                tooltip=[
                    "time",
                    "vehicle_id",
                    "zone",
                    "temperature",
                    "humidity",
                    "fault"
                ]
            )
            .properties(height=380)
        )

        st.altair_chart(chart, use_container_width=True)

        with st.expander("Show anomalous readings"):
            display_columns = [
                "time",
                "vehicle_id",
                "zone",
                "temperature",
                "humidity",
                "fault"
            ]

            display_columns = [
                column for column in display_columns
                if column in df_ml.columns
            ]

            st.dataframe(
                df_ml[df_ml["is_anomaly"]][display_columns].head(50),
                use_container_width=True
            )

    st.divider()

    st.subheader("Latest Sensor Readings")

    st.dataframe(
        df.tail(50),
        use_container_width=True
    )

    if auto_refresh:
        time.sleep(10)
        st.cache_data.clear()
        st.rerun()


if __name__ == "__main__":
    main()