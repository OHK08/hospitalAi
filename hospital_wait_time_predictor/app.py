from flask import Flask, render_template, request, redirect, url_for, flash
import joblib
import pandas as pd
from dateutil import parser
import os

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "your-secret-key"  # Required for flash messages

MODEL_PATH = "model.pkl"

# Load model pipeline at startup
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Place your .pkl here.")

model = joblib.load(MODEL_PATH)

# Helper functions for season and time-of-day
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"

def get_time_of_day(hour):
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"

# Handle Chrome DevTools request to suppress 404
@app.route('/.well-known/appspecific/com.chrome.devtools.json')
def chrome_devtools():
    return '', 204  # No Content response

@app.route("/", methods=["GET"])
def homepage():
    return render_template("homepage.html")

@app.route("/form", methods=["GET"])
def formpage():
    hospitals = [
        "Springfield General Hospital",
        "Riverside Medical Center",
        "Northside Community Hospital",
        "St. Mary’s Regional Health",
        "Summit Health Center"
    ]
    urgencies = ["Low", "Medium", "High"]
    return render_template("form.html", hospitals=hospitals, urgencies=urgencies)

@app.route("/tips", methods=["GET"])
def tipspage():
    return render_template("tips.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        hospital_name = request.form.get("hospital_name")
        urgency = request.form.get("urgency_level")
        date_str = request.form.get("visit_date")
        nurse_ratio = float(request.form.get("nurse_ratio"))
        facility_size = int(request.form.get("facility_size"))

        # Validate inputs
        if not all([hospital_name, urgency, date_str, nurse_ratio, facility_size]):
            raise ValueError("All form fields are required.")

        # Parse date robustly
        dt = parser.parse(date_str)

        input_df = pd.DataFrame([{
            "Hospital Name": hospital_name,
            "Urgency Level": urgency,
            "Day of Week": dt.strftime("%A"),
            "Season": get_season(dt.month),
            "Time of Day": get_time_of_day(dt.hour),
            "Nurse-to-Patient Ratio": nurse_ratio,
            "Facility Size (Beds)": facility_size
        }])

        pred = model.predict(input_df)
        wait_time = float(pred[0])
        wait_time_rounded = round(wait_time, 1)

        return render_template("results.html",
                              prediction=wait_time_rounded,
                              hospital=hospital_name,
                              urgency=urgency,
                              visit_date=date_str,
                              nurse_ratio=nurse_ratio,
                              facility_size=facility_size)
    except Exception as e:
        flash(f"Error: {str(e)}", "danger")
        return redirect(url_for("formpage"))

if __name__ == "__main__":
    app.run(debug=True)