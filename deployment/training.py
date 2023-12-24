import os
import pickle
import argparse
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction import DictVectorizer
from mlflow.sklearn import autolog

# Enable MLflow autologging
autolog()

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("nyc-extra-trees-hyperopt-v1")

def data_preprocess(filename):
    # Check file format and read DataFrame accordingly
    if filename.endswith('.csv'):
        df = pd.read_csv(filename)
        
        # Convert datetime columns to pandas datetime objects
        df.tpep_dropoff_datetime = pd.to_datetime(df.tpep_dropoff_datetime)
        df.tpep_pickup_datetime = pd.to_datetime(df.tpep_pickup_datetime)
        
    elif filename.endswith('.parquet'):
        df = pd.read_parquet(filename)
        # https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month}.parquet
        # df = pd.read_parquet(f"https:nyc/{year}-{month}")

        
    # Calculate trip duration in minutes
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    
    # Filter trips based on duration (between 1 and 60 minutes)
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    
    # Convert selected columns to string type for categorical representation
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    # Create a new column 'PU_DO' by combining pickup and dropoff location IDs
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    
    # Return the processed DataFrame
    return df

def train_and_evaluate_model(train_file, validate_file, model_output_filepath):
    # Preprocess the training data
    df_train = data_preprocess(train_file).head(1000)

    # Preprocess the validation data
    df_validate = data_preprocess(validate_file).head(1000)

    # Print the lengths of the training and validation DataFrames
    print(f"Number of rows in df_train: {len(df_train)}")
    print(f"Number of rows in df_validate: {len(df_validate)}")

    # Load your data (X_train, y_train, X_val, y_val)
    # Define categorical and numerical features: independent features
    categorical = ['PU_DO']
    numerical = ['trip_distance']

    # Initialize a DictVectorizer
    dv = DictVectorizer()

    # Convert training data to a dictionary of records and then transform it into a sparse matrix
    train_dicts = df_train[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)

    # Convert validation data to a dictionary of records and then transform it into a sparse matrix
    val_dicts = df_validate[categorical + numerical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)

    # Dependent feature/target variable
    target = 'duration'
    y_train = df_train[target].values
    y_val = df_validate[target].values

    # Create ExtraTreesRegressor model
    et_model = ExtraTreesRegressor(n_estimators=100, random_state=42)

    # Fit the model
    et_model.fit(X_train, y_train)

    # Predict on validation set
    y_pred = et_model.predict(X_val)

    # Calculate RMSE
    rmse = mean_squared_error(y_val, y_pred, squared=False)

    # Log model and parameters in MLflow
    with mlflow.start_run():
        mlflow.set_tag("model", "ExtraTreesRegressor")
        mlflow.log_params({'n_estimators': 100, 'random_state': 42})
        # Add other hyperparameters as needed
        mlflow.log_metric("rmse", rmse)

        # Save the trained model as a pickle file
        with open(f"{model_output_filepath}/ExtraTreesRegressor_model_v2.pkl", "wb") as f:
            pickle.dump((dv, et_model), f)

        # Register the model in MLflow
        mlflow.sklearn.log_model(et_model, "et_model")
        mlflow.log_artifact(f"{model_output_filepath}/extra_trees_model.pkl")

        print("Training successfully completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Extra Trees model and log metrics to MLflow.')
    parser.add_argument('--year', type=str, help='Year of the data files', required=True)
    parser.add_argument('--month_train', type=str, help='Month of the training data file', required=True)
    parser.add_argument('--month_validate', type=str, help='Month of the validation data file', required=True)
    #parser.add_argument('--model_output_filepath', type=str, help='Path to save the trained model', required=True)

    args = parser.parse_args()

    # Construct file paths based on year and month
    
    model_output_filepath = '/Users/shreyajaiswal/Library/CloudStorage/OneDrive-NortheasternUniversity/Projects/MLOps-NYC-Taxi/experiment_tracking/models'
    train_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{args.year}-{args.month_train}.parquet'
    validate_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{args.year}-{args.month_validate}.parquet'
    # Train and evaluate the model
    train_and_evaluate_model(train_file, validate_file, model_output_filepath)
