import pickle
from flask import Flask, request, jsonify, render_template
import Orange
import pandas as pd
import numpy as np
import os
import traceback
import random

app = Flask(__name__)

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "model", "Final_Project.pkcls")
DATA_FILE = os.path.join(BASE_DIR, "model", "data.csv")

model = None
data_df = None

# --- 1. Load Model (Orange) ---
if os.path.exists(MODEL_FILE):
    try:
        model = pickle.load(open(MODEL_FILE, "rb"))
        print(f"✅ Model loaded successfully from: {MODEL_FILE}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
else:
    print(f"❌ Error: Model file missing at {MODEL_FILE}")

# --- 2. Load Data (Pandas) ---
if os.path.exists(DATA_FILE):
    try:
        # Read CSV as Strings to preserve formatting
        df = pd.read_csv(DATA_FILE, dtype=str)
        # If using raw dataset, we don't need to skip rows. 
        data_df = df
        
        # Clean column names
        data_df.columns = data_df.columns.str.strip()
        
        print(f"✅ Data loaded via Pandas: {len(data_df)} exercises")
    except Exception as e:
        print(f"❌ Error loading CSV with Pandas: {e}")
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
        
        # --- SEARCH & PREDICT ---
        if data_df is not None:
            try:
                # 1. Filter DataFrame (Database Search)
                # We use .copy() to work with a clean subset of data
                matches = data_df[
                    (data_df['Level'] == data.get('Level')) &
                    (data_df['BodyPart'] == data.get('BodyPart')) &
                    (data_df['Type'] == data.get('Type'))
                ].copy()

                # Optional Equipment Filter
                user_equip = data.get('Equipment')
                if user_equip and user_equip != "":
                    matches = matches[matches['Equipment'] == user_equip]

                # 2. Process Matches (Row by Row)
                for _, row in matches.iterrows():
                    
                    # --- Extract Text Details ---
                    title = str(row.get("Title", "Unknown Title"))
                    equip = str(row.get("Equipment", "-"))
                    
                    # Handle Description (Check 'New Desc' then 'Desc')
                    desc = str(row.get("New Desc", ""))
                    if desc == "" or desc.lower() == "nan":
                        desc = str(row.get("Desc", "Description not available."))
                    if desc.lower() == "nan": desc = "-"

                    # --- DETERMINE RATING ---
                    final_rating = "N/A"
                    
                    # Priority A: Check Real Data in CSV
                    real_rating = str(row.get("Rating", "nan"))
                    
                    if real_rating.lower() not in ["nan", "n/a", "?", ""]:
                        final_rating = real_rating
                    else:
                        # Priority B: Use AI Prediction (If data is missing)
                        try:
                            # Build Vector from THIS SPECIFIC ROW'S DATA
                            input_vector = [0] * len(domain.attributes)
                            
                            for i, attr in enumerate(domain.attributes):
                                attr_name = attr.name
                                
                                # Handle One-Hot Attributes (e.g. Equipment=Dumbbell)
                                if "=" in attr_name:
                                    fname, fval = attr_name.split("=", 1)
                                    # Check if the ROW has this specific value
                                    # This ensures 'Equipment' is correctly passed to the AI
                                    if str(row.get(fname)) == fval:
                                        input_vector[i] = 1
                                # Handle Numeric Attributes
                                else:
                                    try:
                                        input_vector[i] = float(row.get(attr_name, 0))
                                    except:
                                        pass
                            
                            # Add placeholders for Target/Metas
                            input_vector.extend([Orange.data.Unknown] * len(domain.class_vars))
                            if hasattr(domain, 'metas'):
                                input_vector.extend([Orange.data.Unknown] * len(domain.metas))

                            # Predict
                            instance = Orange.data.Instance(domain, input_vector)
                            pred_idx = model(instance)
                            idx = int(pred_idx[0]) if isinstance(pred_idx, np.ndarray) else int(pred_idx)
                            final_rating = f"{domain.class_var.values[idx]} (AI Predicted)"
                        except Exception as ai_error:
                            print(f"AI Prediction failed for {title}: {ai_error}")
                            final_rating = "N/A"

                    found_exercises.append({
                        "Title": title,
                        "Equipment": equip,
                        "Description": desc,
                        "Rating": final_rating
                    })

            except Exception as e:
                print(f"⚠️ Error processing matches: {e}")
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
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)