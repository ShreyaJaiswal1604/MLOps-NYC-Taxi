


## Todo

DUE : 23 Dec 23 | 3 pm est
1. Move main.py and req.txt file in this path
2. Create a new Docker file to build the image for the fast apoi container. Follow as similar approach referring to streamlit however the EXPOSE <port> and the CMD to start fastapi server would be different
3. In the docker compose file, under streamlit service, add a new service for fastapi container. 
4. Temp -- copy the model pickle file within the fastapi container for now. Later we can extend it to refer the model from mlflow registry.
5. Minimal changes the fastapi main.py to refer to the model path relative to path which is inside the container. like /app/model/best_model.pkl
6. Check the reponse from STreamlit and do unit testing

Completed on 12/24/2023 -- 01:34 hrs
