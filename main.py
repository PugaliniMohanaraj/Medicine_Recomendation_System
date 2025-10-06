from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime

app = Flask(__name__, template_folder='templates', static_folder='static')

# ---- Inject current_year into all templates (used by footer/headers) ----
@app.context_processor
def inject_now():
    return {"current_year": datetime.now().year}

# ===== Load data =====
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv("datasets/medications.csv")
diets = pd.read_csv("datasets/diets.csv")

# ===== Load model =====
svc = pickle.load(open("models/svc.pkl", "rb"))

# ===== Helpers =====
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]  # list with one row array

    med = medications[medications['Disease'] == dis]['Medication']
    med = [m for m in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [d for d in die.values]

    wrkout = workout[workout['disease'] == dis]['workout']

    return desc, pre, med, die, wrkout


# ===== Vocab / labels =====
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}


# ===== Prediction helper =====
def get_predicted_value(patient_symptoms):
    """
    patient_symptoms: list[str] of raw tokens from the form (e.g., ["headache", "high fever"])
    Returns: disease name (str)
    """
    def norm(s: str) -> str:
        return s.strip().lower().replace(" ", "_")

    vocab_size = len(symptoms_dict)

    # Optional sanity check: make sure model and vocab align
    try:
        if hasattr(svc, "n_features_in_") and svc.n_features_in_ != vocab_size:
            raise ValueError(
                f"Model expects {svc.n_features_in_} features but symptoms_dict has {vocab_size}. "
                "Re-check your symptoms_dict and the training pipeline."
            )
    except Exception:
        pass

    # Build a single-row feature matrix (1 x N)
    X = np.zeros((1, vocab_size), dtype=int)

    for item in patient_symptoms:
        key = norm(item)
        if key in symptoms_dict:
            X[0, symptoms_dict[key]] = 1

    pred_idx = int(svc.predict(X)[0])
    return diseases_list[pred_idx]


# ===== Routes =====
@app.route("/")
def index():
    return render_template("index.html")

# Optional redirects for legacy/bad links
@app.route("/index")
def index_redirect():
    return redirect(url_for("index"), code=301)

@app.route("/templates/index.html")
def tpl_index_redirect():
    return redirect(url_for("index"), code=301)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        symptoms = request.form.get("symptoms", "").strip()

        if not symptoms or symptoms.lower() == "symptoms":
            message = (
                "Please type your symptoms (comma-separated) or use the mic, "
                "and make sure they are spelled correctly."
            )
            return render_template("index.html", message=message)

        # Split and clean user input into tokens
        user_symptoms = [s.strip() for s in symptoms.split(",") if s.strip()]

        # Check if at least one token matches your vocab to avoid empty vectors
        normalized = [s.lower().replace(" ", "_") for s in user_symptoms]
        recognized = [s for s in normalized if s in symptoms_dict]

        if not recognized:
            message = (
                "Sorry, I couldn't recognize any of those symptoms. "
                "Try common symptoms like: headache, high fever, cough, nausea (comma-separated)."
            )
            return render_template("index.html", message=message)

        try:
            predicted_disease = get_predicted_value(user_symptoms)
        except ValueError as e:
            return render_template("index.html", message=f"Model input error: {e}")
        except Exception as e:
            return render_template("index.html", message=f"Unexpected error while predicting: {e}")

        dis_des, pre_list, med_list, rec_diet, wrkout = helper(predicted_disease)

        my_precautions = []
        if pre_list and len(pre_list[0]) > 0:
            for i in pre_list[0]:
                my_precautions.append(i)

        return render_template(
            "index.html",
            predicted_disease=predicted_disease,
            dis_des=dis_des,
            my_precautions=my_precautions,
            medications=med_list,
            my_diet=rec_diet,
            workout=wrkout
        )

    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/developer")
def developer():
    return render_template("developer.html")

@app.route("/blog")
def blog():
    return render_template("blog.html")

# --- Simple search over diseases & symptoms ---
@app.route('/search')
def search():
    q = (request.args.get("q") or "").strip().lower()
    results = []

    if q:
        # diseases
        for idx, name in diseases_list.items():
            if q in name.lower():
                results.append({"type": "Disease", "name": name})
        # symptoms
        for name in symptoms_dict.keys():
            if q in name.lower():
                results.append({"type": "Symptom", "name": name.replace("_", " ")})

    message = None if results else f'No results found for "{q}".'
    return render_template("search_results.html", query=q, results=results, message=message)

# --- Login route (endpoint name = function name => "login") ---
DEMO_USERS = {"demo@health.center": "demo123"}

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = (request.form.get("password") or "").strip()
        if DEMO_USERS.get(email) == password:
            return redirect(url_for("index"))
        return render_template("login.html", message="Invalid email or password.")
    return render_template("login.html")


# Debug helper to confirm routes
@app.route("/__routes")
def __routes():
    lines = []
    for rule in sorted(app.url_map.iter_rules(), key=lambda r: r.rule):
        lines.append(f"{rule.rule}  ->  {rule.endpoint}")
    return "<pre>" + "\n".join(lines) + "</pre>"


if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    app.run(debug=True)
