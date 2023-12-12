import os
import time
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException, Response, status, Query, Request
from fastapi.responses import HTMLResponse
import random

from pydantic import BaseModel


categorical = ['PUlocationID', 'DOlocationID']
numerical = ['trip_distance']

model_path = '/Users/shreyajaiswal/Library/CloudStorage/OneDrive-NortheasternUniversity/Projects/MLOps-NYC-Taxi/experiment_tracking/models/extra_trees_model_v1.pkl'

with open(f'{model_path}', 'rb') as f_out:
    dv,model = pickle.load(f_out)



class inputFeatures(BaseModel):
    pickup_loc: str
    drop_loc: str 
    distance: float

def prepare_features(f: inputFeatures):
    features={}
    features['PU_DO'] = f'{f.pickup_loc}_{f.drop_loc}'
    features['trip_distance'] = f.distance
    return features

def predict_duration(features):
    X = dv.transform(features)
    preds = model.predict(X)
    return float(preds[0])

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}

@app.post('/predict')
async def predict(f : inputFeatures ):
    features = prepare_features(f)
    predicted_duration = predict_duration(features)

    return {
        'duration': predicted_duration
    }





# app = FastAPI()

# @app.get("/")
# async def read_items():
#     return {"message": "Hello World"}

# @app.post("/predict")
# async def predict(tp : TripPlan):
#     """TODO:
#     1. Load the model bin file
#     2. transform the input into vectors
#     3. Use the model to predict the time taken
#     4. Return the result in a JSON format / Dict
#     """
#     return tp


