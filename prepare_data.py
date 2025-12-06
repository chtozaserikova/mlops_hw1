import pandas as pd
from sklearn.datasets import load_iris
import os

def create_dataset():
    os.makedirs("data", exist_ok=True)
    print("Generating Iris dataset...")
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    output_path = "data/train.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")

if __name__ == "__main__":
    create_dataset()