# MLOps Lab – FastAPI and Flask Experiment

This repository contains experiments demonstrating API development and basic ML model serving using **FastAPI** and **Flask**.

## Project Structure

fast_api_experiment/

* **main.py**
  FastAPI application that loads a trained ML model and provides prediction endpoints.

* **train.py**
  Script used to train the machine learning model and save it as `model.pkl`.

* **model.pkl**
  Serialized trained Logistic Regression model.

* **flask_app.py**
  Simple Flask API example with a hello endpoint.

* **fastapi_hello.py**
  Basic FastAPI example demonstrating a simple GET request.

* **requirements.txt**
  List of Python dependencies required to run the experiment.

---

## API Endpoints

### FastAPI ML API

* `/` → Root endpoint describing the API
* `/predict` → POST endpoint used to generate predictions from the trained model
* `/health` → GET endpoint used to check if the API and model are running

### FastAPI Hello API

* `/hello` → Simple GET endpoint returning a greeting message

### Flask API

* `/hello` → Simple Flask GET endpoint returning a greeting message

---

## Model Training

The model is trained using the **Iris dataset** in `train.py`.

Steps performed during training:

1. Load dataset using **scikit-learn**
2. Split dataset into training and testing sets
3. Train a **Logistic Regression** model
4. Save the trained model as `model.pkl`

The FastAPI application (`main.py`) loads this trained model and serves predictions through the `/predict` endpoint.

---

## Technologies Used

* Python
* FastAPI
* Flask
* Uvicorn
* Scikit-learn
* NumPy
