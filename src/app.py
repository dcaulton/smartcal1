#!/usr/bin/env python3.11
"""
smartcal1: Weather-triggered camera test agent (detection2-style)
- Cron every 30min, hostPath /mnt/data/agents/smartcal1
- Ollama phi3:mini for reasoning
- SQLite tasks/weather_logs in /data
- MLFlow logging
- OpenWeatherMap tool
"""
import os
import sqlite3
import requests
import argparse
from datetime import datetime, timedelta
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import mlflow

load_dotenv()  # Local .env support

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="check", choices=["check", "snooze"])
parser.add_argument("--task_id", type=int)
parser.add_argument("--duration", type=str)  # 1d, 2h
args = parser.parse_args()

# Config (k8s env vars override .env)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("MODEL", "phi3:mini")
MLFLOW_URI = os.getenv("MLFLOW_URI", "http://localhost:5000")
WEATHER_API_URL = os.getenv("WEATHER_API_URL")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
LOCATION = os.getenv("LOCATION", "Park Forest,IL,US")
TEMP_THRESHOLD = float(os.getenv("TEMP_THRESHOLD", 50))
DURATION_CHECKS = int(os.getenv("DURATION_CHECKS", 4))  # 2hrs @30min
DB_PATH = os.getenv("DB_PATH", "/data/smartcal.db")

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("/smartcal1")  # MLFlow path-style

# Ollama LLM
llm = OllamaLLM(model=MODEL, base_url=OLLAMA_URL)

# DB Setup (idempotent)
conn = sqlite3.connect(DB_PATH)
conn.executescript("""
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    description TEXT NOT NULL,
    status TEXT DEFAULT 'pending',
    snooze_until DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS weather_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    temp REAL,
    condition_met INTEGER DEFAULT 0
);
""")
conn.commit()

def get_weather():
    """Tool: Fetch current temp"""
    resp = requests.get(
        f"{WEATHER_API_URL}",
        params={"q": LOCATION, "appid": WEATHER_API_KEY, "units": "imperial"},
        timeout=10
    ).json()
    temp = resp["main"]["temp"]
    conn.execute(
        "INSERT INTO weather_logs (temp) VALUES (?)",
        (temp,)
    )
    conn.commit()
    print(f"Current temp in {LOCATION}: {temp:.1f}°F")
    return temp

def check_sustained_warmth():
    """Check last N logs > threshold"""
    since = datetime.now() - timedelta(minutes=30 * DURATION_CHECKS)
    cursor = conn.execute(
        "SELECT COUNT(*) FROM weather_logs WHERE temp > ? AND timestamp > ?",
        (TEMP_THRESHOLD, since)
    )
    count = cursor.fetchone()[0]
    met = count >= DURATION_CHECKS
    print(f"Sustained >{TEMP_THRESHOLD}°F for {count}/{DURATION_CHECKS} checks: {met}")
    return met

def create_task(llm_reasoning=""):
    """Create pending task"""
    description = f"Test outside camera setup (reasoning: {llm_reasoning or 'weather trigger'})"
    cursor = conn.execute(
        "INSERT INTO tasks (description) VALUES (?) RETURNING id",
        (description,)
    )
    task_id = cursor.fetchone()[0]
    conn.commit()
    print(f"✅ Created task #{task_id}: {description}")
    return task_id

def send_reminder(task_id):
    """Placeholder: Discord/log reminder"""
    cursor = conn.execute("SELECT description FROM tasks WHERE id=?", (task_id,))
    desc = cursor.fetchone()[0]
    print(f"🚨 REMINDER: Task #{task_id}\n{desc}")
    # Future: requests.post(DISCORD_WEBHOOK_URL, json={"content": f"Task {task_id}"})
    return True

def snooze_task(task_id, duration_str):
    """Snooze logic"""
    now = datetime.now()
    if duration_str.endswith("d"):
        delta = timedelta(days=int(duration_str[:-1]))
    elif duration_str.endswith("h"):
        delta = timedelta(hours=int(duration_str[:-1]))
    else:
        delta = timedelta(days=1)
    snooze_until = now + delta
    conn.execute(
        "UPDATE tasks SET status='snoozed', snooze_until=? WHERE id=?",
        (snooze_until, task_id)
    )
    conn.commit()
    print(f"⏳ Snoozed task #{task_id} until {snooze_until}")

# Agent Reasoning Loop (simple state machine → LangGraph later)
if args.mode == "check":
    with mlflow.start_run(run_name=f"check-{datetime.now().isoformat()}"):
        mlflow.log_param("location", LOCATION)
        mlflow.log_param("temp_threshold", TEMP_THRESHOLD)
        mlflow.log_param("duration_checks", DURATION_CHECKS)

        temp = get_weather()
        mlflow.log_metric("current_temp", temp)

        if check_sustained_warmth():
            # LLM reasoning
            prompt = f"""
            Weather in {LOCATION}: {temp}°F sustained >{TEMP_THRESHOLD}°F for 2+ hrs.
            Should we remind to test outside camera? Reason briefly, confirm Y/N.
            """
            reasoning = llm.invoke(prompt).strip()
            mlflow.log_text("llm_reasoning", reasoning)

            if "Y" in reasoning.upper() or "YES" in reasoning.upper():
                task_id = create_task(reasoning)
                send_reminder(task_id)
                mlflow.log_metric("tasks_created", 1)
                mlflow.log_metric("reminders_sent", 1)
            else:
                print(f"LLM declined: {reasoning}")
                mlflow.log_metric("tasks_created", 0)
        else:
            mlflow.log_metric("tasks_created", 0)

        # Report pending tasks
        cursor = conn.execute(
            "SELECT id, status, description FROM tasks WHERE status IN ('pending', 'snoozed') ORDER BY created_at DESC LIMIT 5"
        )
        tasks = cursor.fetchall()
        if tasks:
            print("\n📋 The List:")
            for tid, status, desc in tasks:
                print(f"  #{tid} [{status}] {desc}")

elif args.mode == "snooze" and args.task_id and args.duration:
    snooze_task(args.task_id, args.duration)

conn.close()
