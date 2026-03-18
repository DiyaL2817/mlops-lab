from sklearn.datasets import load_diabetes
import pandas as pd
import os

os.makedirs("data", exist_ok=True)

data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

df.to_csv("data/diabetes.csv", index=False)

print("Dataset saved")