import os 
import joblib
import pandas as pd 
import numpy as np 
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

MODEL_FILE = 'model.pkl'
PIPELINE_FILE = 'pipeline.pkl'

def build_pipeline(num_attributes, cat_attributes):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy = "median")),     
        ("scaler", StandardScaler())
    ])

    # Contructing a pipeline for Categorical columns
    cat_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    #  Contructing a full pipeline
    full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attributes),
            ('cat', cat_pipeline, cat_attributes)
    ])
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    housing = pd.read_csv('housing.csv')

    # Create a Stratified Test set
    housing ['income_cat'] = pd.cut(housing['median_income'] , 
                                    bins= [0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                                    labels = [1, 2, 3, 4, 5])

    split = StratifiedShuffleSplit(n_splits = 1 , test_size = 0.2 , random_state = 42)

    for train_index, test_index in split.split(housing , housing['income_cat']):
        housing.loc[test_index].drop('income_cat', axis =1).to_csv('input.csv', index = False)
        housing = housing.loc[train_index].drop('income_cat', axis =1)

    housing_labels = housing['median_house_value'].copy()
    housing_features = housing.drop('median_house_value' , axis = 1)

    # print(housing, housing_labels)

    # 5. Separate Numerical and Categorical columns
    num_attributes = housing_features.drop('ocean_proximity' , axis =1).columns.tolist()
    cat_attributes = ['ocean_proximity']

    pipeline = build_pipeline(num_attributes, cat_attributes)
    housing_prepared = pipeline.fit_transform(housing_features)

    model = RandomForestRegressor()
    model.fit(housing_prepared, housing_labels)

    # SAVE MODEL AND THE PIPELINE
    joblib.dump(model,MODEL_FILE)
    joblib.dump(pipeline,PIPELINE_FILE)
    print("Model Trained and saved to file")

else :
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    input_data = pd.read_csv('input.csv')
    transformed_data = pipeline.transform(input_data)
    predictions = model.predict(transformed_data)
    input_data['median_house_value'] = predictions
    
    input_data.to_csv('output.csv', index = False)
    print("Inference Complete. Results saved to output.csv file")
