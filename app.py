from flask import Flask
import subprocess

app = Flask(__name__)

@app.route("/")
def home():
    return "Web Intrusion Detection System Running 🚀"

@app.route("/detect")
def detect():
    result = subprocess.run(["python", "iso.py"], capture_output=True, text=True)
    return f"<pre>{result.stdout}</pre>"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)