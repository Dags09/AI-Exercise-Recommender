# 🏋️ AI Exercise Recommender

A machine learning–powered web application that recommends exercises based on a user's **fitness level**, **target body part**, **exercise type**, and **available equipment**. Built to solve "workout paralysis" — the common problem of gym-goers not knowing which exercises best match their goals and skill level.

🔗 **Live Demo:** [ai-exerciserecommender.onrender.com](https://ai-exerciserecommender.onrender.com/)

---

## 📖 Overview

Beginners and intermediate athletes often struggle to choose the right exercises for their skill level and target muscle group. This project provides instant, data-driven exercise recommendations by predicting an exercise's effectiveness (rating) using a trained classification model.

**In scope:**

- Recommendations for gym exercises, bodyweight movements, and cardio activities
- Coverage of major muscle groups (Chest, Legs, Arms, etc.)
- Coverage of common equipment types (Dumbbells, Barbells, Machines, etc.)

**Out of scope:**

- Video demonstrations
- Nutrition planning
- Injury diagnosis

---

## 🧠 Model Details

| Attribute                    | Description                                                                                                                  |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **Algorithm**                | Random Forest Classifier                                                                                                     |
| **Why this model?**          | Handles categorical data well (body part, level, equipment) and is more resistant to overfitting than a single decision tree |
| **Input Features**           | `Level`, `BodyPart`, `Type`, `Equipment`                                                                                     |
| **Target Variable**          | `Rating` (discretized into 3 classes: **Bad**, **Good**, **Great**)                                                          |
| **Trees**                    | 50                                                                                                                           |
| **Max Tree Depth**           | 20                                                                                                                           |
| **Min Subset Size to Split** | 20                                                                                                                           |

### Why discretize the rating?

The original `Rating` variable had 10 distinct numerical classes, which caused overfitting and low predictive precision during training. Reclassifying it into 3 broader classes (Bad, Good, Great) significantly improved model stability and precision.

---

## 📊 Dataset

- **Source:** [MegaGymDataset](https://www.kaggle.com/) (Kaggle)
- **Type:** Tabular (text/categorical) data
- **Samples:** 2,919
- **Features:** 4 (`Level`, `BodyPart`, `Type`, `Equipment`)
- **Target:** `Rating`

### Preprocessing Pipeline

1. **Imputation** — Missing `Rating` values filled using the "Average/Most Frequent" strategy
2. **Discretization** — `Rating` converted from a continuous value into 3 categorical bins:
    - `< 5.75` → **Bad**
    - `5.75 – 8.45` → **Good**
    - `≥ 8.45` → **Best**
3. **Missing Description Handling** — Empty `Desc` fields replaced with a `"None"` placeholder via a custom formula
4. **One-Hot Encoding** — Categorical features (`Level`, `BodyPart`, `Type`, `Equipment`) converted into numeric vectors
5. **No explicit normalization/standardization** applied, since Random Forest is robust to unscaled categorical data

> Preprocessing and training were performed using **Orange Data Mining**, with the pipeline: `File → Impute → Select Rows → Discretize → Edit Domain → Formula → Select Columns → Data Sampler → Random Forest → Test and Score → Confusion Matrix`

---

## 📈 Evaluation Results

Cross-validated (5-fold, stratified) performance per class:

| Class     | Accuracy | Precision | Recall | F1    | AUC   |
| --------- | -------- | --------- | ------ | ----- | ----- |
| **Bad**   | 0.669    | 0.495     | 0.352  | 0.412 | 0.615 |
| **Good**  | 0.600    | 0.395     | 0.372  | 0.383 | 0.562 |
| **Great** | 0.566    | 0.394     | 0.526  | 0.450 | 0.591 |

**Confusion Matrix (sum of probabilities):**

|                  | Predicted: Bad | Predicted: Good | Predicted: Best | Total |
| ---------------- | -------------- | --------------- | --------------- | ----- |
| **Actual: Bad**  | 286.6          | 243.9           | 235.5           | 766   |
| **Actual: Good** | 247.2          | 270.0           | 262.9           | 780   |
| **Actual: Best** | 237.4          | 263.9           | 287.7           | 789   |

---

## 🛠️ Tech Stack

- **Model Training:** Orange Data Mining (Random Forest)
- **Backend:** Flask + Orange3 (loads the trained `.pkcls` model directly for inference)
- **Data Handling:** Pandas
- **Frontend:** HTML/CSS (Jinja templates)
- **Hosting:** [Render](https://render.com/)
- **Dataset:** MegaGymDataset (Kaggle)

### How predictions work

1. The trained Orange model (`Final_Project.pkcls`) and the source `data.csv` are loaded once at startup.
2. When a user submits `Level`, `BodyPart`, `Type`, and (optionally) `Equipment` via `POST /predict`, the backend filters `data.csv` for matching exercises.
3. For each matching exercise:
    - If the dataset already has a real `Rating`, that value is returned directly.
    - If the `Rating` is missing, the backend builds a one-hot feature vector from that row and asks the Random Forest model for a live prediction, returned as e.g. `"Great (AI Predicted)"`.
4. The API responds with a JSON list of matching exercises, each including `Title`, `Equipment`, `Description`, and `Rating`.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip

Your `requirements.txt`:

```
Flask
gunicorn
orange3
numpy
scipy
scikit-learn
pandas
PyQt5
```

### Installation

```bash
git clone https://github.com/<your-username>/ai-exercise-recommender.git
cd ai-exercise-recommender
pip install -r requirements.txt
```

### Running Locally

```bash
python app.py
```

Then open `http://localhost:5000` in your browser.

### Production Deployment

The app is served in production via **Gunicorn** (as used on Render):

```bash
gunicorn app:app
```

### API Usage

**Endpoint:** `POST /predict`

**Request body:**

```json
{
    "Level": "Beginner",
    "BodyPart": "Chest",
    "Type": "Strength",
    "Equipment": "Dumbbell"
}
```

_(`Equipment` is optional — omit or leave empty to search across all equipment types.)_

**Response:**

```json
{
    "count": 2,
    "exercises": [
        {
            "Title": "Dumbbell Bench Press",
            "Equipment": "Dumbbell",
            "Description": "Lie on a flat bench...",
            "Rating": "Great"
        },
        {
            "Title": "Dumbbell Fly",
            "Equipment": "Dumbbell",
            "Description": "Lie on a flat bench...",
            "Rating": "Good (AI Predicted)"
        }
    ]
}
```

---

## 📁 Project Structure

```
ai-exercise-recommender/
├── app.py                    # Flask app: loads model + data, serves /predict endpoint
├── model/
│   ├── Final_Project.pkcls   # Trained Random Forest model (Orange3)
│   └── data.csv              # Source exercise dataset
├── templates/
│   └── index.html            # Main frontend page
├── static/
│   └── style.css              # Styling
├── requirements.txt
└── README.md
```

---

## 🔮 Future Work

- **Expand the dataset** with more user reviews to balance classes and improve F1-scores for the "Good" and "Great" categories
- **NLP on exercise descriptions** to enable goal-based recommendations (e.g., "back pain relief") instead of category-only matching
- **User accounts** for personalized workout history tracking and feedback-driven learning
- **Visual media integration** — GIFs or video demonstrations for each recommended exercise

---

## 👥 Authors

Ateinza — submitted in partial fulfilment of the requirements for **ITE102: Artificial Intelligence**, College of Information & Computing Sciences.
