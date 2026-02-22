from flask import Flask, render_template, request
import sqlite3
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import math
from datetime import datetime

# -------------------------------
# Create Flask App
# -------------------------------
app = Flask(__name__)

# -------------------------------
# Create Database Tables
# -------------------------------
def create_table():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    # Reports table with confidence and risk
    c.execute("""
    CREATE TABLE IF NOT EXISTS reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image TEXT,
        latitude REAL,
        longitude REAL,
        address TEXT,
        prediction TEXT,
        created_at TEXT,
        verified INTEGER DEFAULT 0,
        confidence REAL DEFAULT 0,
        risk TEXT DEFAULT ''
    )
    """)
    
    # Alerts table (added species column for solution feature)
    c.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        latitude REAL,
        longitude REAL,
        severity TEXT,
        message TEXT,
        created_at TEXT,
        status TEXT,
        species TEXT
    )
    """)

    # Satellite anomaly table
    c.execute("""
    CREATE TABLE IF NOT EXISTS anomalies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        latitude REAL,
        longitude REAL,
        radius REAL,
        description TEXT
    )
    """)

    conn.commit()
    conn.close()

# Call once at startup
create_table()

# -------------------------------
# Load ML Model
# -------------------------------
model = load_model("model/model.h5")

classes = [
    "Dung beetle",
    "Apple snail",
    "Asian longhorned beetle"
]

# -------------------------------
# Prediction Function
# -------------------------------
def predict_species(image_path):
    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    class_index = np.argmax(prediction)
    confidence = float(prediction[class_index])

    if confidence < 0.70:
        return "Unknown Species", confidence
    else:
        return classes[class_index], confidence

# -------------------------------
# Distance Calculation (Haversine)
# -------------------------------
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in KM
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = (math.sin(dLat/2) ** 2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(dLon/2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# -------------------------------
# Spread Trend Analysis
# -------------------------------
def get_spread_trend():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("""
        SELECT COUNT(*) FROM reports
        WHERE prediction IN ('Apple snail', 'Asian longhorned beetle')
        AND datetime(created_at) >= datetime('now', '-7 days')
    """)
    current_week = c.fetchone()[0]

    c.execute("""
        SELECT COUNT(*) FROM reports
        WHERE prediction IN ('Apple snail', 'Asian longhorned beetle')
        AND datetime(created_at) BETWEEN datetime('now', '-14 days') AND datetime('now', '-7 days')
    """)
    previous_week = c.fetchone()[0]

    conn.close()

    if current_week > previous_week:
        trend = "Spreading 🔺"
    elif current_week == previous_week:
        trend = "Stable ➖"
    else:
        trend = "Declining 🔻"

    return current_week, previous_week, trend

# -------------------------------
# Home Route
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -------------------------------
# Upload Route
# -------------------------------
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["image"]
    latitude = float(request.form["latitude"])
    longitude = float(request.form["longitude"])
    address = request.form["address"]

    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    result, confidence = predict_species(filepath)
    invasive_species = ["Apple snail", "Asian longhorned beetle"]

    # Default risk
    if result == "Unknown Species":
        risk = "Uncertain - Low Confidence Detection"
    elif result in invasive_species:
        risk = "High Risk (Invasive Species Detected)"
    else:
        risk = "Low Risk (Native Species)"

    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    cluster_count = 0
    recent_count = 0
    anomaly_flag = 0

    if result in invasive_species:
        c.execute("""
            SELECT latitude, longitude, created_at
            FROM reports
            WHERE prediction IN ('Apple snail', 'Asian longhorned beetle')
            AND datetime(created_at) >= datetime('now', '-7 days')
        """)
        invasive_reports = c.fetchall()
        recent_count = len(invasive_reports)

        for report in invasive_reports:
            dist = calculate_distance(latitude, longitude, float(report[0]), float(report[1]))
            if dist <= 20:
                cluster_count += 1

        if cluster_count >= 2:
            c.execute("""
                INSERT INTO anomalies (latitude, longitude, radius, description)
                VALUES (?, ?, ?, ?)
            """, (latitude, longitude, 20000,
                  "Vegetation Stress Detected via Satellite Correlation"))
            anomaly_flag = 1

        # Adjusted risk calculation
        risk_score_value = (cluster_count * 0.4) + (recent_count * 0.3) + (anomaly_flag * 0.3)

        if risk_score_value < 3:
            risk = "Low"
        elif risk_score_value < 6:
            risk = "Moderate"
        elif risk_score_value < 9:
            risk = "High"
        else:
            risk = "Critical 🚨"

        # Insert alert for High or Critical risks, include species column
        if risk in ["High", "Critical 🚨"]:
            c.execute("""
                INSERT INTO alerts (latitude, longitude, severity, message, created_at, status, species)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (latitude, longitude, risk,
                  f"{result} detected at high risk",
                  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                  "Active",
                  result))

    # Save report with confidence and risk
    c.execute("""
        INSERT INTO reports (image, latitude, longitude, address, prediction, created_at, confidence, risk)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (file.filename, latitude, longitude, address, result,
          datetime.now().strftime("%Y-%m-%d %H:%M:%S"), confidence, risk))

    conn.commit()
    conn.close()

    return render_template("result.html",
                           image=file.filename,
                           prediction=result,
                           risk=risk,
                           latitude=latitude,
                           longitude=longitude,
                           address=address,
                           confidence=round(confidence * 100, 2))

# -------------------------------
# Map Route
# -------------------------------
@app.route("/map")
def map_view():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("SELECT latitude, longitude, prediction FROM reports")
    reports = c.fetchall()

    c.execute("SELECT latitude, longitude, radius, description FROM anomalies")
    anomalies = c.fetchall()

    conn.close()
    return render_template("map.html", reports=reports, anomalies=anomalies)

# -------------------------------
# Dashboard Route
# -------------------------------
@app.route("/dashboard")
def dashboard():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM reports")
    total_reports = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM reports WHERE prediction IN ('Apple snail', 'Asian longhorned beetle')")
    invasive_count = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM reports WHERE prediction = 'Dung beetle'")
    native_count = c.fetchone()[0]

    # Active alerts
    c.execute("SELECT COUNT(*) FROM alerts WHERE status='Active'")
    active_alerts = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM reports WHERE verified=0")
    unverified_reports = c.fetchone()[0]

    c.execute("SELECT id FROM reports ORDER BY created_at DESC LIMIT 1")
    last_report = c.fetchone()
    last_report_id = last_report[0] if last_report else None

    conn.close()
    current_week, previous_week, trend = get_spread_trend()

    return render_template("dashboard.html",
                           total=total_reports,
                           invasive=invasive_count,
                           native=native_count,
                           active_alerts=active_alerts,
                           unverified_reports=unverified_reports,
                           current_week=current_week,
                           previous_week=previous_week,
                           trend=trend,
                           last_report_id=last_report_id)

# -------------------------------
# Alerts Dashboard Route (Updated)
# -------------------------------
@app.route("/alerts")
def alerts():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    
    c.execute("SELECT * FROM alerts ORDER BY created_at DESC")
    alerts_data = c.fetchall()
    conn.close()

    species_solutions = {
        "Apple snail": "Drain water and remove snails manually.Wear gloves and dispose properly.Monitor the area weekly for new snails.Consider introducing natural predators if appropriate",
        "Asian longhorned beetle": "Cut and burn infested trees.Inspect nearby trees for eggs.Notify forestry authorities.Maintain quarantine measures.Monitor for regrowth or new infestations."
    }

    invasive_list = ["Apple snail", "Asian longhorned beetle"]

    alerts_list = []
    for alert in alerts_data:
        id = alert[0]
        lat = alert[1]
        lng = alert[2]
        severity = alert[3]
        message = alert[4]
        created_at = alert[5]
        status = alert[6]
        
        # If species column exists, use it. Otherwise, extract from message
        species_name = alert[7] if len(alert) > 7 and alert[7] else message.split(" ")[0]

        alerts_list.append({
            "id": id,
            "latitude": lat,
            "longitude": lng,
            "severity": severity,
            "message": message,
            "created_at": created_at,
            "status": status,
            "is_invasive": species_name in invasive_list,
            "solution": species_solutions.get(species_name, "No solution available")
        })

    return render_template("alerts.html", alerts=alerts_list)

# -------------------------------
# Admin Review, Approve/Reject, Result Routes (unchanged)
# -------------------------------
@app.route("/admin/review")
def admin_review():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("""
        SELECT id, image, latitude, longitude, address, prediction, created_at
        FROM reports
        WHERE verified=0
        ORDER BY created_at DESC
    """)
    reports = c.fetchall()
    conn.close()
    return render_template("admin_review.html", reports=reports)

@app.route("/admin/approve/<int:report_id>")
def approve_report(report_id):
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("UPDATE reports SET verified=1 WHERE id=?", (report_id,))
    conn.commit()
    conn.close()
    return "Report Approved! <a href='/admin/review'>Back to Review</a>"

@app.route("/admin/reject/<int:report_id>")
def reject_report(report_id):
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("UPDATE reports SET verified=-1 WHERE id=?", (report_id,))
    conn.commit()
    conn.close()
    return "Report Rejected! <a href='/admin/review'>Back to Review</a>"

@app.route("/result/<int:report_id>")
def view_result(report_id):
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("""
        SELECT image, latitude, longitude, address, prediction, confidence, risk
        FROM reports
        WHERE id=?
    """, (report_id,))
    report = c.fetchone()
    conn.close()

    if not report:
        return "Report not found"

    image = report[0]
    latitude = report[1]
    longitude = report[2]
    address = report[3]
    prediction = report[4]
    confidence = round(report[5] * 100, 2)
    risk = report[6]

    return render_template("result.html",
                           image=image,
                           prediction=prediction,
                           risk=risk,
                           latitude=latitude,
                           longitude=longitude,
                           address=address,
                           confidence=confidence)

# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)