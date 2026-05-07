import logging

from sklearn.ensemble import IsolationForest


log = logging.getLogger(__name__)

MIN_SAMPLES = 50


def detect_anomalies(df, contamination=0.05):

    if df.empty or len(df) < MIN_SAMPLES:
        log.warning(
            "Not enough data for anomaly detection. Rows available: %d, required: %d",
            len(df),
            MIN_SAMPLES
        )
        return df

    required_columns = ["temperature", "humidity"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column for ML: {column}")

    df = df.copy()

    features = df[required_columns].values

    model = IsolationForest(
        contamination=contamination,
        random_state=42
    )

    df["anomaly_score"] = model.fit_predict(features)
    df["is_anomaly"] = df["anomaly_score"] == -1

    anomaly_count = int(df["is_anomaly"].sum())

    log.info(
        "Isolation Forest detected %d anomalies from %d records (%.1f%%)",
        anomaly_count,
        len(df),
        100 * anomaly_count / len(df)
    )

    return df