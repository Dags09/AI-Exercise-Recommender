import pickle
from flask import Flask, request, jsonify, render_template
import Orange
import numpy as np
import os
import traceback
import random

# --- PATH CONFIGURATION ---

# 1. Get the folder where THIS file (app.py) lives
#    This is: .../YourProject/templates
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Fix: project root should be the folder where app.py lives (CURRENT_DIR) ---
# 2. Define the Project Root (Go up one level)
#    This is: .../YourProject
# PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
# 2. Define the Project Root as the folder that contains app.py
#    (so model/ and static/ live under the same folder as app.py)
PROJECT_ROOT = CURRENT_DIR

# 3. Configure Flask to look in the Root for HTML, and Root/static for CSS
# app = Flask(__name__, 
#             template_folder=PROJECT_ROOT,    # index.html is in the root
#             static_folder=os.path.join(PROJECT_ROOT, 'static')) # css is in static/
app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_ROOT, 'templates'),
    static_folder=os.path.join(PROJECT_ROOT, 'static'),
)

# 4. Define paths for Model and Data (Inside the 'model' folder in Root)
MODEL_FILE = os.path.join(PROJECT_ROOT, "model", "Final_Project.pkcls")
DATA_FILE = os.path.join(PROJECT_ROOT, "model", "data.csv")

model = None
data_table = None

# --- Load Model & Data ---
if os.path.exists(MODEL_FILE):
    try:
        model = pickle.load(open(MODEL_FILE, "rb"))
        print(f"✅ Model loaded from: {MODEL_FILE}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
else:
    print(f"❌ Error: Model file missing at {MODEL_FILE}")

if os.path.exists(DATA_FILE):
    try:
        data_table = Orange.data.Table(DATA_FILE)
        print(f"✅ Data loaded from: {DATA_FILE} ({len(data_table)} rows)")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
else:
    print(f"⚠️ Warning: Data file missing at {DATA_FILE}")


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded.'}), 500

    try:
        data = request.get_json()
        domain = model.domain
        
        found_exercises = []
        
        # --- 1. SEARCH FOR MATCHING EXERCISES ---
        if data_table:
            try:
                matches = []
                for row in data_table:
                    r_level = str(row["Level"])
                    r_body = str(row["BodyPart"])
                    r_type = str(row["Type"])
                    r_equip = str(row["Equipment"])

                    # Filter Logic
                    if (r_level == data.get('Level') and 
                        r_body == data.get('BodyPart') and 
                        r_type == data.get('Type')):
                        
                        # Optional Equipment Filter
                        user_equip = data.get('Equipment')
                        if user_equip and user_equip != "" and user_equip != r_equip:
                            continue 
                        
                        matches.append(row)

                # --- 2. PREDICT RATING FOR EACH MATCH ---
                for row in matches:
                    # A. Build Input Vector
                    input_vector = [0] * len(domain.attributes)

                    for i, attr in enumerate(domain.attributes):
                        attr_name = attr.name
                        
                        if "=" in attr_name:
                            feature_name, feature_value = attr_name.split("=", 1)
                            row_val = str(row[feature_name]) if feature_name in row.domain else None
                            if row_val == feature_value:
                                input_vector[i] = 1
                        else:
                            try:
                                input_vector[i] = float(row[attr_name])
                            except:
                                pass 

                    # Add placeholders
                    input_vector.extend([Orange.data.Unknown] * len(domain.class_vars))
                    if hasattr(domain, 'metas'):
                        input_vector.extend([Orange.data.Unknown] * len(domain.metas))

                    # B. Run Prediction
                    instance = Orange.data.Instance(domain, input_vector)
                    prediction_idx = model(instance)
                    
                    if isinstance(prediction_idx, np.ndarray):
                        idx = int(prediction_idx[0])
                    else:
                        idx = int(prediction_idx)
                    
                    individual_rating = domain.class_var.values[idx]

                    # C. Get Text Details
                    try:
                        title = str(row["Title"])
                        equip = str(row["Equipment"])
                        
                        desc = "None"
                        try:
                            desc = str(row["New Desc"])
                        except:
                            try:
                                desc = str(row["Desc"])
                            except:
                                desc = "Description not available."
                        
                        if desc.strip() == "" or desc.lower() == "nan":
                            desc = "None"

                        found_exercises.append({
                            "Title": title,
                            "Equipment": equip,
                            "Description": desc,
                            "Rating": individual_rating
                        })
                    except Exception as e:
                        print(f"Skipping row error: {e}")

            except Exception as e:
                print(f"⚠️ Error searching/predicting: {e}")
                traceback.print_exc()

        return jsonify({
            'count': len(found_exercises),
            'exercises': found_exercises
        })

    except Exception as e:
        print(f"🔥 Critical Error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)