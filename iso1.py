import pandas as pd
import numpy as np
import re
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

LOG_FILE = "sample_logs.txt"


# --------------------------------------------------
# Log Parser
# --------------------------------------------------

def parse_log_line(line):
    pattern = r'(\d+\.\d+\.\d+\.\d+) - - \[(.*?)\] "GET (.*?) HTTP/\d\.\d" (\d+) (\d+) "(.*?)" "(.*?)"'
    match = re.match(pattern, line.strip())
    if match:
        return {
            "raw": line.strip(),
            "ip": match.group(1),
            "request": match.group(3),
            "status": int(match.group(4)),
            "bytes": int(match.group(5)),
            "user_agent": match.group(7)
        }
    return None


def load_logs(filepath):
    data = []
    with open(filepath, "r") as f:
        for line in f:
            parsed = parse_log_line(line)
            if parsed:
                data.append(parsed)
    return pd.DataFrame(data)


# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------

def extract_features(df):
    df_features = df.copy()

    ip_freq = df_features['ip'].value_counts()
    df_features['ip_freq'] = df_features['ip'].map(ip_freq)

    unique_urls = df_features.groupby('ip')['request'].nunique()
    df_features['unique_url_count'] = df_features['ip'].map(unique_urls)

    total_bytes = df_features.groupby('ip')['bytes'].sum()
    df_features['ip_total_bytes'] = df_features['ip'].map(total_bytes)

    df_features['status_flag'] = df_features['status'].apply(
        lambda x: 1 if x in [401, 403, 404, 500] else 0
    )

    feature_cols = [
        'ip_freq',
        'unique_url_count',
        'ip_total_bytes',
        'status_flag'
    ]

    return df_features[feature_cols], df_features


# --------------------------------------------------
# MAIN DETECTION
# --------------------------------------------------

def main():
    print("="*70)
    print("LIVE CLOUD ANOMALY DETECTION")
    print("="*70)

    df = load_logs(LOG_FILE)

    if df.empty:
        print("No logs available.")
        return

    X, df_features = extract_features(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.01,
        random_state=42
    )

    model.fit(X_scaled)
    y_pred = model.predict(X_scaled)

    anomalies = np.sum(y_pred == -1)
    total_logs = len(y_pred)
    percentage = (anomalies / total_logs) * 100

    print(f"\nTotal Logs Analysed: {total_logs}")
    print(f"Anomalies Detected: {anomalies}")
    print(f"Anomaly Percentage: {percentage:.2f}%")

    print("\nSample Anomalous Logs:")
    anomalous_logs = df[y_pred == -1]['raw'].head(5)

    for i, log in enumerate(anomalous_logs):
        print(f"{i+1}. {log}")

    print("\nDetection Completed.")
    print("="*70)


if __name__ == "__main__":
    main()