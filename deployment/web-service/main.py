import os
import time
import pickle
import pandas as pd
#from google.cloud import storage
from fastapi import FastAPI, HTTPException, Response, status, Query, Request
from fastapi.responses import HTMLResponse
import random

categorical = ['PUlocationID', 'DOlocationID']
numerical = ['trip_distance']

model_path = '/Users/shreyajaiswal/Library/CloudStorage/OneDrive-NortheasternUniversity/Projects/MLOps-NYC-Taxi/experiment_tracking/models/extra_trees_model.pkl'


# with open(f"{model_path}", 'rb') as f_in:
#     dv,lr = pickle.load

# description = """
# API helps to provide an ML model as a service

# ## Users will be able to:
#     ***Predict Taxi Ride Duration***
# """

# tags_metadata = [{
#     "name": "time",
#     "description": "Predicts the mean  duration of ride time between twolocations",
# },
# ]

# app = FastAPI( title = "NYC-Predict taxi ride duration",
#               description=description)


app = FastAPI()

@app.get("/anushka")
async def read_items():
    return {"message": "Hello World - Piyush"}

# # Load the trained model from the pickle file
# with open(f"{model_output_filepath}/extra_trees_model.pkl", "rb") as f:
#     loaded_model = pickle.load(f)

# # Predict on the test set
# y_test_pred = loaded_model.predict(X_test)

# def predict(location1, location2, distance):



@app.post("/shreya")
async def predict(location1, location2, distance):
    # your logic
    result = random.random()
    return {"message": f"Time est: {result}"}
