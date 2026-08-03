import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request

# Create WSGI application
app = Flask(__name__)

# Dynamically resolve file paths for Render environment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "overfitting_model.pkl")


def load_model():
    """Safely load the pickle model."""
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as model_file:
                return pickle.load(model_file), None
        except Exception as e:
            return None, f"Model load error: {str(e)}"
    return (
        None,
        f"File 'overfitting_model.pkl' not found at path: {MODEL_PATH}",
    )


# Glassmorphism dark-theme HTML interface with dynamic animations
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Diabetes Risk Assessment | ML Portal</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
            --glass-bg: rgba(30, 41, 59, 0.7);
            --glass-border: rgba(255, 255, 255, 0.1);
            --primary-accent: #6366f1;
            --primary-hover: #4f46e5;
            --glow-color: rgba(99, 102, 241, 0.25);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --danger-bg: rgba(239, 68, 68, 0.15);
            --danger-border: #ef4444;
            --success-bg: rgba(16, 185, 129, 0.15);
            --success-border: #10b981;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 30px 20px;
        }

        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .card {
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: 24px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            width: 100%;
            max-width: 640px;
            padding: 40px;
            animation: fadeIn 0.8s cubic-bezier(0.16, 1, 0.3, 1);
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .header {
            text-align: center;
            margin-bottom: 32px;
        }

        .header h1 {
            font-size: 1.85rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            background: linear-gradient(to right, #ffffff, #a5b4fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }

        .header p {
            font-size: 0.925rem;
            color: var(--text-muted);
        }

        .grid-layout {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 16px;
        }

        @media (max-width: 580px) {
            .grid-layout {
                grid-template-columns: 1fr;
            }
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        .form-group.full-width {
            grid-column: 1 / -1;
        }

        .form-group label {
            font-size: 0.825rem;
            font-weight: 600;
            margin-bottom: 6px;
            color: var(--text-main);
            letter-spacing: 0.01em;
        }

        .form-control {
            width: 100%;
            padding: 12px 14px;
            border-radius: 10px;
            border: 1px solid var(--glass-border);
            background-color: rgba(15, 23, 42, 0.6);
            color: var(--text-main);
            font-size: 0.9rem;
            transition: all 0.2s ease;
        }

        .form-control:focus {
            outline: none;
            border-color: var(--primary-accent);
            box-shadow: 0 0 0 4px var(--glow-color);
            background-color: rgba(15, 23, 42, 0.85);
        }

        .btn-submit {
            grid-column: 1 / -1;
            width: 100%;
            padding: 14px;
            border: none;
            border-radius: 12px;
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
            color: #ffffff;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-top: 12px;
            box-shadow: 0 4px 12px var(--glow-color);
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(99, 102, 241, 0.4);
        }

        .btn-submit:active {
            transform: translateY(0);
        }

        .result-box {
            margin-top: 28px;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            animation: slideUp 0.5s ease;
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .result-box.positive {
            background-color: var(--danger-bg);
            border: 1px solid var(--danger-border);
        }

        .result-box.negative {
            background-color: var(--success-bg);
            border: 1px solid var(--success-border);
        }

        .result-box.error-box {
            background-color: var(--danger-bg);
            border: 1px solid var(--danger-border);
        }

        .result-title {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-muted);
            margin-bottom: 6px;
        }

        .result-value {
            font-size: 1.25rem;
            font-weight: 700;
        }

        .result-box.positive .result-value { color: #f87171; }
        .result-box.negative .result-value { color: #34d399; }
        .result-box.error-box .result-value { color: #f87171; font-size: 1rem; }
    </style>
</head>
<body>
    <div class="card">
        <div class="header">
            <h1>Diabetes Risk Predictor</h1>
            <p>Enter 8 health parameters for instant AI analysis</p>
        </div>

        <form action="/predict" method="POST" class="grid-layout">
            <div class="form-group">
                <label for="pregnancies">Pregnancies</label>
                <input type="number" step="any" id="pregnancies" name="pregnancies" class="form-control" placeholder="e.g. 2" value="{{ inputs.get('pregnancies', '') }}" required>
            </div>

            <div class="form-group">
                <label for="glucose">Glucose Level</label>
                <input type="number" step="any" id="glucose" name="glucose" class="form-control" placeholder="e.g. 120" value="{{ inputs.get('glucose', '') }}" required>
            </div>

            <div class="form-group">
                <label for="blood_pressure">Blood Pressure (mmHg)</label>
                <input type="number" step="any" id="blood_pressure" name="blood_pressure" class="form-control" placeholder="e.g. 70" value="{{ inputs.get('blood_pressure', '') }}" required>
            </div>

            <div class="form-group">
                <label for="skin_thickness">Skin Thickness (mm)</label>
                <input type="number" step="any" id="skin_thickness" name="skin_thickness" class="form-control" placeholder="e.g. 20" value="{{ inputs.get('skin_thickness', '') }}" required>
            </div>

            <div class="form-group">
                <label for="insulin">Insulin Level (mu U/ml)</label>
                <input type="number" step="any" id="insulin" name="insulin" class="form-control" placeholder="e.g. 79" value="{{ inputs.get('insulin', '') }}" required>
            </div>

            <div class="form-group">
                <label for="bmi">BMI</label>
                <input type="number" step="any" id="bmi" name="bmi" class="form-control" placeholder="e.g. 25.5" value="{{ inputs.get('bmi', '') }}" required>
            </div>

            <div class="form-group">
                <label for="dpf">Diabetes Pedigree Function</label>
                <input type="number" step="any" id="dpf" name="dpf" class="form-control" placeholder="e.g. 0.47" value="{{ inputs.get('dpf', '') }}" required>
            </div>

            <div class="form-group">
                <label for="age">Age (Years)</label>
                <input type="number" step="any" id="age" name="age" class="form-control" placeholder="e.g. 33" value="{{ inputs.get('age', '') }}" required>
            </div>

            <button type="submit" class="btn-submit">Run Prediction</button>
        </form>

        {% if prediction is not none %}
        <div class="result-box {{ 'error-box' if is_error else ('positive' if prediction == 1 else 'negative') }}">
            <div class="result-title">{{ "System Status" if is_error else "Prediction Result" }}</div>
            <div class="result-value">
                {% if is_error %}
                    {{ prediction }}
                {% else %}
                    {{ "High Risk of Diabetes (Class 1)" if prediction == 1 else "Low Risk / Normal (Class 0)" }}
                {% endif %}
            </div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def home():
    model, error = load_model()
    if error:
        return render_template_string(
            HTML_TEMPLATE, prediction=error, is_error=True, inputs={}
        )
    return render_template_string(
        HTML_TEMPLATE, prediction=None, is_error=False, inputs={}
    )


@app.route("/predict", methods=["POST"])
def predict():
    model, error = load_model()

    # Capture form field values to refill form inputs
    form_inputs = {
        "pregnancies": request.form.get("pregnancies", ""),
        "glucose": request.form.get("glucose", ""),
        "blood_pressure": request.form.get("blood_pressure", ""),
        "skin_thickness": request.form.get("skin_thickness", ""),
        "insulin": request.form.get("insulin", ""),
        "bmi": request.form.get("bmi", ""),
        "dpf": request.form.get("dpf", ""),
        "age": request.form.get("age", ""),
    }

    if error:
        return render_template_string(
            HTML_TEMPLATE,
            prediction=error,
            is_error=True,
            inputs=form_inputs,
        )

    try:
        # Extract inputs as float sequence
        features = [
            float(form_inputs["pregnancies"]),
            float(form_inputs["glucose"]),
            float(form_inputs["blood_pressure"]),
            float(form_inputs["skin_thickness"]),
            float(form_inputs["insulin"]),
            float(form_inputs["bmi"]),
            float(form_inputs["dpf"]),
            float(form_inputs["age"]),
        ]

        # Model Inference
        features_array = np.array([features])
        prediction_val = int(model.predict(features_array)[0])

        return render_template_string(
            HTML_TEMPLATE,
            prediction=prediction_val,
            is_error=False,
            inputs=form_inputs,
        )

    except ValueError:
        return render_template_string(
            HTML_TEMPLATE,
            prediction="Input Error: Please fill in valid numeric values for all parameters.",
            is_error=True,
            inputs=form_inputs,
        )
    except Exception as e:
        return render_template_string(
            HTML_TEMPLATE,
            prediction=f"Inference Error: {str(e)}",
            is_error=True,
            inputs=form_inputs,
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
