## Import Libraries
import numpy as np
from fastapi import FastAPI, Form, HTTPException
import joblib
import os
from utils import process_new
from dotenv import load_dotenv


## Load the dotenv file
_ = load_dotenv(override=True)
secret_key = os.getenv('BACKEND_KEY_TOKEN')

## Load the model
model = joblib.load('models/forest_model_with_smote.pkl')


## Initilalize an app
app = FastAPI(debug=True)


## The Function for Prediction
@app.post('/predict_churn_classification')
async def predict_churn(KeyToken: str=Form(...), 
                        CreditScore: float=Form(...), 
                        Geography: str=Form(..., description='Geography', enum=['France', 'Spain', 'Germany']),
                        Gender: str=Form(..., description='Gender', enum=['Male', 'Female']), 
                        Age: int=Form(...), 
                        Tenure: int=Form(..., description='Tenure', enum=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 
                        Balance: float=Form(...), 
                        NumOfProducts: int=Form(..., description='NumOfProducts', enum=[0, 1]), 
                        HasCrCard: int=Form(..., description='HasCrCard', enum=[0, 1]),
                        IsActiveMember: int=Form(..., description='IsActiveMember', enum=[0, 1]),
                        EstimatedSalary: float=Form(...)):

    ## Validation
    if KeyToken not in [secret_key]:
        raise HTTPException(status_code=403, detail='You are notauthorized to use this API.')

    ## Concatenate all Feats
    X_new = [CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]
    X_new = np.array(X_new)

    ## Call the custom function
    X_processed = process_new(X_new=X_new)

    ## Predict using the model
    y_pred = model.predict(X_processed)[0]

    ## To get True of False
    churn_pred = bool(y_pred)
    
    return {f'Churn Classification is: {churn_pred}'}