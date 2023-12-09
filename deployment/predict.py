import argparse
import os
import pickle
import pandas as pd

categorical = ['PUlocationID', 'DOlocationID']
numerical = ['trip_distance']

model_path = '/Users/shreyajaiswal/Library/CloudStorage/OneDrive-NortheasternUniversity/Projects/MLOps-NYC-Taxi/experiment_tracking/models/extra_trees_model.pkl'


with open(f"{model_path}", 'rb') as f_in:
    dv,lr = pickle.load


def read_data(year:str, month:str):
    df = pd.read_parquet(f'/Users/shreyajaiswal/Library/CloudStorage/OneDrive-NortheasternUniversity/Projects/MLOps-NYC-Taxi/experiment_tracking/dataset/yellow_tripdata_{year}_{month}.parquet')
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

    return df

def main(year:str, month:str):
    df = read_data(year, month)
    dicts = df[categorical + numerical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    print(f'Mean predict value for the year {year} and month {month} is {y_pred.mean():.2f}')


if __name__ == 'main':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--year",
        default="2022",
        type= str,
        help = "year for which prediction needs to be done, example 2022"
    )

    parser.add_argument(
        "--month",
        default="01",
        type=str,
        help="month number, example 01 for January"
    )

    args = parser.parse_args()

    if(len(args.year) != 4) or (len(args.month)!=2):
        print("Please enter valid input values, Example: --year 2022 --month 01")
        exit(101)

    main(args.year, args.month)


    
