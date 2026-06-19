import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import pickle

def train(df):

    # keep only required columns
    df = df[["brand", "year", "fuel_type", "km_driven", "price"]]

    # create mappings (IMPORTANT)
    brand_map = {k: v for v, k in enumerate(df["brand"].astype("category").cat.categories)}
    fuel_map = {k: v for v, k in enumerate(df["fuel_type"].astype("category").cat.categories)}

    df["brand"] = df["brand"].map(brand_map)
    df["fuel_type"] = df["fuel_type"].map(fuel_map)

    X = df.drop("price", axis=1)
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor()
    }

    scores = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        scores[name] = model.score(X_test, y_test)

    best_model_name = max(scores, key=scores.get)
    best_model = models[best_model_name]

    # SAVE EVERYTHING (IMPORTANT)
    with open("best_model.pkl", "wb") as f:
        pickle.dump({
            "model": best_model,
            "brand_mapping": brand_map,
            "fuel_mapping": fuel_map
        }, f)

    return scores, best_model_name