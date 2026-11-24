# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Ingest from the Adzuna API (DACH)
# MAGIC
# MAGIC Pulls live data-job postings for **Germany, Austria and
# MAGIC Switzerland** and caches every raw response verbatim to a Unity
# MAGIC Catalog volume.
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md ## Configuration

# COMMAND ----------

CAT    = "jobs"
BRONZE = f"{CAT}.bronze"
VOL    = f"/Volumes/{CAT}/bronze/raw"

BASE = "https://api.adzuna.com/v1/api/jobs"

COUNTRIES = ["at"]

ROLES = [

    # PRIORITY 1: Robotics
    "robotics engineer",
    "robotics software engineer",
    "robotics research engineer",
    "robot software developer",
    "mobile robotics",
    "robotik entwickler",
    "robotik ingenieur",
    "robotik softwareentwickler",
    "ROS2",

    # PRIORITY 1: SLAM and autonomous systems
    "SLAM",
    "SLAM engineer",
    "autonomous systems",
    "autonomous driving",
    "localization engineer",
    "navigation engineer",
    "path planning",
    "motion planning",
    "visual odometry",
    "autonome systeme",
    "autonomes fahren",
    "pfadplanung",
    "Lokalisierung",

    # PRIORITY 1: Sensors and perception
    "sensor fusion",
    "sensor engineer",
    "sensor development",
    "perception engineer",
    "LiDAR",
    "sensorfusion",
    "sensorik",
    "sensorentwickler",
    "sensortechnik",

    # PRIORITY 1: Computer vision
    "computer vision",
    "computer vision engineer",
    "image processing",
    "machine vision",
    "bildverarbeitung",
    "bilderkennung",
    "3D vision",

    # PRIORITY 1: Drones and UAV
    "drone engineer",
    "UAV engineer",
    "flight control engineer",
    "drone development",
    "drohnenentwicklung",

    # PRIORITY 2: Mechatronics
    "mechatronics engineer",
    "mechatronics development",
    "system engineer mechatronics",
    "mechatronik",
    "mechatronik entwickler",
    "entwicklungsingenieur",

    # PRIORITY 2: Control engineering
    "control systems engineer",
    "control engineer",
    "controls software engineer",
    "regelungstechnik",
    "regelungsingenieur",
    "steuerungstechnik",

    # PRIORITY 2: MATLAB and simulation
    "MATLAB",
    "Simulink",
    "simulation engineer",
    "model based development",
    "model based systems engineering",
    "simulationsingenieur",
    "modellentwicklung",
    "modellbasierte entwicklung",

    # PRIORITY 2: Automotive
    "automotive development engineer",
    "ADAS engineer",
    "vehicle dynamics engineer",
    "functional safety engineer",
    "fahrzeugentwicklung",
    "fahrerassistenzsysteme",

    # PRIORITY 3: FEM and structural mechanics
    "finite element analysis",
    "FEA engineer",
    "FEM engineer",
    "Abaqus",
    "Code_Aster",
    "berechnungsingenieur",
    "FEM simulation",
    "strukturmechanik",
    "festigkeitsberechnung",

    # PRIORITY 3: Digital twins
    "digital twin",
    "digital twin engineer",
    "digitaler zwilling",
    "virtual validation",

    # Additional: Automation and testing
    "automation engineer",
    "automatisierungstechnik",
    "automatisierungsingenieur",
    "test engineer",
    "validation engineer",
    "validierungsingenieur",

    # Additional: Research and graduate opportunities
    "research engineer",
    "research assistant robotics",
    "PhD robotics",
    "PhD mechatronics",
    "doctoral researcher robotics",
    "forschungsingenieur",
    "wissenschaftlicher mitarbeiter",
    "universitaetsassistent",
    "praedoc",
    "trainee engineering",
    "graduate engineer",
]

MAX_PAGES = 2         

# COMMAND ----------

# MAGIC %md ## Credentials 
# MAGIC

# COMMAND ----------

import json, pathlib

_c = json.loads(pathlib.Path(
    f"/Volumes/{CAT}/bronze/conf/adzuna.json").read_text())

APP_ID, APP_KEY = _c["app_id"], _c["app_key"]
assert APP_ID and APP_KEY, "credentials file is empty"

# COMMAND ----------

# MAGIC %md ## Verify the country endpoints
# MAGIC

# COMMAND ----------

import requests

for c in COUNTRIES:
    r = requests.get(
        f"{BASE}/{c}/search/1",
        params={"app_id": APP_ID, "app_key": APP_KEY,
                "results_per_page": 5, "what": ROLES[0],
                "content-type": "application/json"}, timeout=30)
    print(f"{c}: HTTP {r.status_code}  "
          f"{r.json().get('count') if r.ok else r.text[:80]}")

# COMMAND ----------

# MAGIC %md ## Pulling the actual data

# COMMAND ----------

import time, datetime, random

RETRIES = 7      # 7 attempts with the backoff below spans ~2 minutes
PACE    = 1.0    # seconds between successful calls

def backoff(attempt):
     return min(30, 2 ** attempt) * (0.5 + random.random())

TODAY = datetime.date.today().isoformat()
pathlib.Path(VOL).mkdir(parents=True, exist_ok=True)

calls_made = 0
cache_hits = 0


def slug(s):
    return s.replace(" ", "-")


def fetch(country, role, page):
    global calls_made, cache_hits

    fname = f"{country}__{slug(role)}__p{page}__{TODAY}.json"
    path = pathlib.Path(VOL) / fname

    if path.exists():
        cache_hits += 1
        return json.loads(path.read_text())

    params = {
        "app_id": APP_ID,
        "app_key": APP_KEY,
        "results_per_page": 50,
        "what": role,
        "content-type": "application/json",
    }

 
    RETRY_STATUS = {429, 500, 502, 503, 504}

    for attempt in range(RETRIES):
        try:
            r = requests.get(f"{BASE}/{country}/search/{page}",
                             params=params, timeout=30)
        except requests.exceptions.RequestException as e:
            # connection reset / read timeout - same treatment
            wait = backoff(attempt)
            print(f"  {type(e).__name__}, sleeping {wait:.1f}s")
            time.sleep(wait)
            continue

        if r.status_code in RETRY_STATUS:
            # honour Retry-After when the server sends one
            ra = r.headers.get("Retry-After")
            wait = float(ra) if ra and ra.isdigit() else backoff(attempt)
            print(f"  {r.status_code}, sleeping {wait:.1f}s "
                  f"(attempt {attempt + 1}/{RETRIES})")
            time.sleep(wait)
            continue

        if r.status_code == 410:                 # past the last page
            return {"results": [], "count": 0}

        r.raise_for_status()
        path.write_text(r.text)                  # cache before parsing
        calls_made += 1
        time.sleep(PACE)                         # be polite
        return r.json()

    raise RuntimeError(
        f"gave up on {country}/{role}/p{page} after {RETRIES} attempts")


# COMMAND ----------

# MAGIC %md
# MAGIC ##Load

# COMMAND ----------

summary = []

for country in COUNTRIES:
    for role in ROLES:
        for page in range(1, MAX_PAGES + 1):
            data  = fetch(country, role, page)
            n     = len(data.get("results", []))
            total = data.get("count", 0)

            if page == 1:
                summary.append((country, role, total))
                print(f"{country}  {role:28} "
                      f"total={total:6} page1={n}")

            if n < 50:               
                break

print(f"\nAPI calls made this run : {calls_made}")
print(f"Served from cache       : {cache_hits}")
