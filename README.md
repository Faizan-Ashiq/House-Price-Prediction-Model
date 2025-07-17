# House-Price-Prediction-Model
This project is a pure Data Scientiist project in which I predict the price of a house from a data

This repository presents a complete machine learning pipeline designed to predict California housing prices using a Random Forest model. The pipeline handles all stages of the ML workflow — from data loading and preprocessing to model training, evaluation, and making predictions on new data.

---

##  Table of Contents

- [Project Objective](#project-objective)
- [Key Features](#key-features)
- [Dataset Description](#dataset-description)
- [Technologies Used](#technologies-used)
- [Pipeline Breakdown](#pipeline-breakdown)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Outputs](#outputs)
- [Future Improvements](#future-improvements)
- [Contact](#contact)

---

##  Project Objective

The goal of this project is to:
- Build an end-to-end machine learning pipeline that can be used in real-world housing price prediction.
- Automate preprocessing of both numerical and categorical features.
- Save trained models and reuse them without re-training.
- Apply proper model evaluation and testing methods (Stratified Sampling).
- Use joblib and inferences at last to secure all the data

---

##  Key Features

###  1. Data Preprocessing Pipelines
- **Numerical Features**: Missing values are imputed using the median strategy and then scaled using `StandardScaler`.
- **Categorical Features**: Encoded using `OneHotEncoder` with unknown value handling for robustness.

###  2. Stratified Sampling
- Data is split into training and test sets using **StratifiedShuffleSplit** to maintain the proportion of income categories.

###  3. Modular Code Structure
- Clean separation between pipeline building, model training, and inference stages.

###  4. Model Reusability
- Trained model and transformation pipeline are saved to disk using `joblib`, so re-training is not required unless the model file is removed.

###  5. Automated Inference
- If the model already exists, the script skips training and directly loads the model to predict values from new input (`input.csv`) and saves predictions to `output.csv`.

---

##  Dataset Description

The dataset is based on **California housing data** and includes:
- Median income
- House age
- Number of rooms and bedrooms
- Population
- Proximity to the ocean (`ocean_proximity`)
- Target variable: `median_house_value`

Input File: `housing.csv`  
Test Samples Saved As: `input.csv`  
Prediction Output File: `output.csv`

---

##  Technologies Used

| Component         | Tools / Libraries         |
|-------------------|---------------------------|
| Programming       | Python                    |
| Data Handling     | pandas, numpy             |
| Machine Learning  | scikit-learn              |
| Serialization     | joblib                    |
| Model Used        | RandomForestRegressor     |
| Preprocessing     | Pipelines, ColumnTransformer |

---

## ⚙ Pipeline Breakdown

1. **Check if the model is already trained**  
   - If not, it reads `housing.csv`, applies preprocessing, trains the model, and saves everything.
   - If already trained, it skips training and loads `model.pkl` and `pipeline.pkl`.

2. **Create Stratified Train-Test Split**  
   - Ensures income levels are fairly represented using the `income_cat` feature.

3. **Feature Engineering**  
   - Separates numerical and categorical features.
   - Constructs transformation pipelines using `Pipeline` and `ColumnTransformer`.

4. **Model Training**  
   - Trains `RandomForestRegressor` on the transformed features.
   - Saves both the model and the pipeline using `joblib`.

5. **Inference Phase**  
   - Reads `input.csv`, applies saved pipeline, performs predictions, and saves results to `output.csv`.

---

## ▶ How to Run

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
