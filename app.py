import pickle
from flask import Flask, request, jsonify, render_template
import Orange
import numpy as np
import os
import traceback
import random

# --- STANDARD FLASK SETUP ---
app = Flask(__name__)

# --- PATH CONFIGURATION ---
# Get the folder where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths to your model folder
MODEL_FILE = os.path.join(BASE_DIR, "model", "Final_Project.pkcls")
DATA_FILE = os.path.join(BASE_DIR, "model", "data.csv")

model = None
data_table = None

if os.path.exists(MODEL_FILE):
    try:
        model = pickle.load(open(MODEL_FILE, "rb"))
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

if os.path.exists(DATA_FILE):
    try:
        # Load data using Orange
        data_table = Orange.data.Table(DATA_FILE)
        print(f"✅ Training Data loaded from {DATA_FILE} ({len(data_table)} rows)")
    except Exception as e:
        print(f"❌ Error loading CSV data: {e}")
else:
    print(f"⚠️ WARNING: {DATA_FILE} not found.")

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
        
        if data_table:
            try:
                # Find rows that match user input
                matches = []
                for row in data_table:
                    r_level = str(row["Level"])
                    r_body = str(row["BodyPart"])
                    r_type = str(row["Type"])

                    # Basic Filtering
                    if (r_level == data.get('Level') and 
                        r_body == data.get('BodyPart') and 
                        r_type == data.get('Type')):
                        
                        # Equipment Filtering (Optional)
                        user_equip = data.get('Equipment')
                        r_equip = str(row["Equipment"])
                        if user_equip and user_equip != "" and user_equip != r_equip:
                            continue 
                        
                        matches.append(row)

                # --- PROCESS EACH MATCH ---
                for row in matches:
                    title = str(row["Title"])
                    equip = str(row["Equipment"])
                    
                    # 1. Get Description
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

                    # 2. GET RATING (The Fix)
                    # First, try to read the ACTUAL rating from the CSV
                    final_rating = "N/A"
                    try:
                        actual_rating = str(row["Rating"])
                        # Only use it if it's valid (not missing, not '?')
                        if actual_rating and actual_rating != "?" and actual_rating.lower() != "nan":
                            final_rating = actual_rating
                    except:
                        pass
                    
                    # 3. IF NO ACTUAL RATING, USE AI PREDICTION
                    if final_rating == "N/A" or final_rating == "?":
                        try:
                            # Build Input Vector for this specific row
                            input_vector = [0] * len(domain.attributes)
                            for i, attr in enumerate(domain.attributes):
                                attr_name = attr.name
                                if "=" in attr_name:
                                    feature_name, feature_value = attr_name.split("=", 1)
                                    # Check the ROW data, not the user input
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
                            
                            # Predict
                            instance = Orange.data.Instance(domain, input_vector)
                            pred_idx = model(instance)
                            idx = int(pred_idx[0]) if isinstance(pred_idx, np.ndarray) else int(pred_idx)
                            final_rating = domain.class_var.values[idx] + " (Predicted)"
                        except:
                            final_rating = "Unknown"

                    # Add to results
                    found_exercises.append({
                        "Title": title,
                        "Equipment": equip,
                        "Description": desc,
                        "Rating": final_rating
                    })

            except Exception as e:
                print(f"⚠️ Error searching: {e}")
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