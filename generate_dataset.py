"""
Generate Web Log Dataset for Isolation Forest (1% Attack Ratio)
"""

import random
from datetime import datetime, timedelta

NORMAL_COUNT = 1980
ATTACK_COUNT = 20
OUTPUT_FILE = "sample_logs.txt"

normal_ips = [f"192.168.1.{i}" for i in range(10, 80)]
attack_ips = ["203.0.113.100"]

normal_endpoints = [
    "/", "/home", "/products", "/about",
    "/blog", "/contact", "/search?q=laptop",
    "/assets/style.css", "/assets/app.js"
]

attack_endpoints = [
    "/../../etc/passwd",
    "/.git/config",
    "/admin/login",
    "/backup.sql",
    "/search?q=' OR 1=1--"
]

normal_user_agents = [
    "Mozilla/5.0 Chrome/120.0",
    "Mozilla/5.0 Firefox/121.0",
    "Mozilla/5.0 Safari/605.1"
]

attack_user_agents = [
    "sqlmap/1.6",
    "Nikto/2.1.6",
    "curl/7.68.0"
]

start_time = datetime(2026, 2, 5, 8, 0, 0)

def generate_timestamp(base_time, offset):
    dt = base_time + timedelta(seconds=offset)
    return dt.strftime("%d/%b/%Y:%H:%M:%S +0000")

def format_log(ip, timestamp, endpoint, status, size, ua):
    return f'{ip} - - [{timestamp}] "GET {endpoint} HTTP/1.1" {status} {size} "-" "{ua}"\n'

ground_truth = []

with open(OUTPUT_FILE, "w") as f:

    # NORMAL TRAFFIC
    for i in range(NORMAL_COUNT):
        ip = random.choice(normal_ips)
        endpoint = random.choice(normal_endpoints)
        status = 200
        size = random.randint(1000, 5000)
        ua = random.choice(normal_user_agents)
        timestamp = generate_timestamp(start_time, i*2)

        line = format_log(ip, timestamp, endpoint, status, size, ua)
        f.write(line)
        ground_truth.append((line, 0))

    # ATTACK TRAFFIC (burst from same IP)
    for i in range(ATTACK_COUNT):
        ip = attack_ips[0]
        endpoint = random.choice(attack_endpoints)
        status = random.choice([401,403,500])
        size = random.randint(200,600)
        ua = random.choice(attack_user_agents)
        timestamp = generate_timestamp(start_time, NORMAL_COUNT*2 + i)

        line = format_log(ip, timestamp, endpoint, status, size, ua)
        f.write(line)
        ground_truth.append((line, 1))

# Save ground truth
with open("ground_truth.txt", "w") as gt:
    for line, label in ground_truth:
        gt.write(f"{label}|{line}")

print("Dataset generated successfully.")
print("Total records:", len(ground_truth))
print("Attack ratio:", ATTACK_COUNT/len(ground_truth))