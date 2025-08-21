from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib
import numpy as np
from chatbot import get_bot_response, format_response

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "secret"

# Load the trained multi-output XGBoost model
model = joblib.load("tuned_multioutput_xgboost.pkl")

# -----------------------------
# Homepage
# -----------------------------
@app.route("/")
def homepage():
    return render_template("homepage.html")


# -----------------------------
# Form Page
# -----------------------------
@app.route("/form")
def formpage():
    # These can be dynamically loaded from dataset if needed
    regions = ["Rural", "Urban"]
    urgencies = ["Low", "Medium", "High", "Critical"]
    return render_template("form.html", regions=regions, urgencies=urgencies)


# -----------------------------
# Predict & Show Result Page
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect input values from form
        data = {
            "Region": [request.form.get("Region")],
            "Day of Week": [request.form.get("Day of Week")],
            "Season": [request.form.get("Season")],
            "Time of Day": [request.form.get("Time of Day")],
            "Urgency Level": [request.form.get("Urgency Level")],
            "Nurse-to-Patient Ratio": [float(request.form.get("Nurse-to-Patient Ratio"))],
            "Specialist Availability": [int(request.form.get("Specialist Availability"))],
            "Facility Size (Beds)": [int(request.form.get("Facility Size (Beds)"))]
        }

        # Convert to DataFrame
        input_df = pd.DataFrame(data)

        # Predict wait times
        predictions = model.predict(input_df)[0]  # [Total, Registration, Triage, Medical Professional]
        predictions = predictions.tolist()  # Convert NumPy array to list for Jinja2 rendering

        # Identify bottleneck step
        bottleneck_steps = ["Registration", "Triage", "Medical Professional"]
        bottleneck_idx = int(np.argmax(predictions[1:4]))  # Index of max wait among sub-stages
        bottleneck = bottleneck_steps[bottleneck_idx]

        # Render prediction page
        return render_template(
            "predict.html",
            predictions=predictions,
            bottleneck=bottleneck,
            error=None
        )

    except Exception as ex:
        return render_template(
            "predict.html",
            predictions=None,
            bottleneck=None,
            error="Error making prediction: " + str(ex)
        )


# -----------------------------
# Chatbot API Endpoint
# -----------------------------
@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.json.get("message")
    bot_reply = get_bot_response(user_input)
    formatted_reply = format_response(bot_reply)
    return jsonify({"response": formatted_reply})


@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")


# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
