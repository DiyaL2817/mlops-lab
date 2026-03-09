from fastapi import FastAPI, HTTPException 
from pydantic import BaseModel
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Train a simple model
data = load_iris()
X, y = data.data[:100], data.target[:100]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)

app = FastAPI(
    title="Logistic Regression API",
    description="A simple API for Logistic Regression predictions",
    version="1.0",
)

class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

@app.post('/predict')
def predict(data: InputData):
    features = np.array([[data.feature1, data.feature2, data.feature3, data.feature4]])
    pred = model.predict(features)
    return {'prediction': int(pred[0])}

@app.get('/health')
def health_check():
    return {'status': 'API is running'}