from flask import Flask, request
import subprocess
import datetime
import random

app = Flask(__name__)

LOG_FILE = "sample_logs.txt"

@app.route("/")
def home():
    return "Web Intrusion Detection System Running"

@app.route("/simulate")
def simulate():
    # Simulate incoming web log
    ip = request.remote_addr
    timestamp = datetime.datetime.utcnow().strftime("%d/%b/%Y:%H:%M:%S +0000")
    endpoint = random.choice(["/", "/admin", "/.git/config", "/search?q=test"])
    status = random.choice([200, 403, 500])
    size = random.randint(200, 4000)
    ua = request.headers.get("User-Agent")

    log_line = f'{ip} - - [{timestamp}] "GET {endpoint} HTTP/1.1" {status} {size} "-" "{ua}"\n'

    with open(LOG_FILE, "a") as f:
        f.write(log_line)

    return "Live log generated"

@app.route("/detect")
def detect():
    result = subprocess.run(["python", "iso1.py"], capture_output=True, text=True)
    return f"<pre>{result.stdout}</pre>"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)