import os
import joblib
import pandas as pd

# Define paths to saved models
BASE_DIR = os.path.dirname(__file__)
REGRESSOR_PATH = os.path.join(BASE_DIR, 'trained_models', 'workload_hours_model.joblib')
CLASSIFIER_PATH = os.path.join(BASE_DIR, 'trained_models', 'burnout_classifier_model.joblib')

# Global cache for loaded models
_regressor = None
_classifier = None

def _get_models():
    """Loads and caches models in memory for fast inference."""
    global _regressor, _classifier
    if _regressor is None or _classifier is None:
        if not os.path.exists(REGRESSOR_PATH) or not os.path.exists(CLASSIFIER_PATH):
            raise FileNotFoundError("Trained models not found. Run 'python ml_engine/train_model.py' first.")
        _regressor = joblib.load(REGRESSOR_PATH)
        _classifier = joblib.load(CLASSIFIER_PATH)
    return _regressor, _classifier

def predict_workload_and_burnout(input_data: dict) -> dict:
    """
    Predict weekly workload from the full academic burden profile rather than capping the value
    to teaching-only thresholds. Burnout risk and weekly working hours must stay consistent.
    """
    regressor, classifier = _get_models()

    df_input = pd.DataFrame([input_data])
    raw_hours = float(regressor.predict(df_input)[0])

    teaching_hours = float(input_data.get('teaching_hours', 0) or 0)
    backlog_days = int(input_data.get('grading_backlog_days', 0) or 0)
    thesis_students = int(input_data.get('supervised_theses', 0) or 0)
    weekend_hours = float(input_data.get('weekend_work_hours', 0) or 0)
    late_night_actions = int(input_data.get('lms_late_night_actions', 0) or 0)
    fatigue = int(input_data.get('self_reported_fatigue', 5) or 5)
    support = int(input_data.get('perceived_support', 3) or 3)

    workload_hours = (
        max(teaching_hours, 8.0)
        + backlog_days * 0.9
        + thesis_students * 2.6
        + weekend_hours * 1.8
        + late_night_actions * 0.5
        + max(0, fatigue - 4) * 1.8
        + max(0, 4 - support) * 2.2
    )

    predicted_category = str(classifier.predict(df_input)[0])
    category_floor = {
        'LOW': 12.0,
        'MODERATE': 22.0,
        'HIGH': 32.0,
        'CRITICAL': 40.0,
    }
    workload_hours = max(workload_hours, category_floor.get(predicted_category, 12.0))
    workload_hours = max(workload_hours, raw_hours)

    # Keep workload hours aligned with risk severity so critical lecturers do not appear under the 40h threshold.
    if predicted_category == 'CRITICAL':
        workload_hours = max(workload_hours, 40.0)
    elif predicted_category == 'HIGH':
        workload_hours = max(workload_hours, 30.0)
    elif predicted_category == 'MODERATE':
        workload_hours = max(workload_hours, 20.0)

    predicted_hours = min(max(workload_hours, 8.0), 60.0)

    probabilities = classifier.predict_proba(df_input)[0]
    class_labels = list(classifier.classes_)
    prob_dict = dict(zip(class_labels, probabilities))

    weights = {'LOW': 20.0, 'MODERATE': 45.0, 'HIGH': 70.0, 'CRITICAL': 90.0}
    burnout_risk_score = sum(prob_dict.get(cat, 0.0) * w for cat, w in weights.items())

    if predicted_category == 'LOW':
        burnout_risk_score = min(burnout_risk_score, 35.0)
    elif predicted_category == 'MODERATE':
        burnout_risk_score = min(max(burnout_risk_score, 36.0), 55.0)
    elif predicted_category == 'HIGH':
        burnout_risk_score = min(max(burnout_risk_score, 56.0), 75.0)
    else:
        burnout_risk_score = max(burnout_risk_score, 75.0)

    recommendation = _generate_recommendation(
        predicted_category,
        predicted_hours,
        input_data.get('grading_backlog_days', 0),
        input_data.get('self_reported_fatigue', 5)
    )

    return {
        'predicted_weekly_hours': round(predicted_hours, 1),
        'burnout_risk_category': predicted_category,
        'burnout_risk_score': round(burnout_risk_score, 1),
        'recommended_action': recommendation
    }

def _generate_recommendation(category: str, hours: float, backlog: int, fatigue: int) -> str:
    """Generates automated actionable advice for HOD and Lecturer."""
    if category == 'CRITICAL':
        return (
            f"CRITICAL OVERLOAD ({hours:.1f} hrs/wk): Immediate HOD intervention required. "
            f"Reassign TA support for grading backlog ({backlog} days) or adjust module allocations."
        )
    elif category == 'HIGH':
        return (
            f"ELEVATED STRAIN ({hours:.1f} hrs/wk): High risk of fatigue buildup. "
            f"Consider reallocating thesis supervision load or postponing non-essential admin duties."
        )
    elif category == 'MODERATE':
        return (
            f"SUSTAINABLE LOAD ({hours:.1f} hrs/wk): Workload is within manageable bounds. "
            f"Monitor upcoming exam grading periods to avoid backlog spikes."
        )
    else:
        return f"OPTIMAL CAPACITY ({hours:.1f} hrs/wk): Lecturer has spare capacity for additional research or module duties."