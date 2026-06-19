import streamlit as st
import pandas as pd
import pickle
from train_model import train

st.title("🚗 Car Price Prediction App")

uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    st.success("Dataset uploaded successfully!")

    # Train model
    scores, best_model_name = train(df)

    st.write("### Model Performance")

    for model_name, score in scores.items():
        st.write(f"{model_name}: {score:.2f}")

    st.success(f"Best Model: {best_model_name}")

    # Load saved model
    with open("best_model.pkl", "rb") as f:
        model_bundle = pickle.load(f)

    model = model_bundle["model"]
    brand_mapping = model_bundle["brand_mapping"]
    fuel_mapping = model_bundle["fuel_mapping"]

    st.write("### Enter Car Details")

    brand = st.selectbox("Select Car Brand", df["brand"].unique())
    year = st.number_input("Enter Year", min_value=1990, max_value=2024, value=2015)
    km_driven = st.number_input("Enter KM Driven", min_value=0, value=50000)
    fuel_type = st.selectbox("Select Fuel Type", df["fuel_type"].unique())

    if st.button("Predict Price"):

        # check mapping
        if brand not in brand_mapping or fuel_type not in fuel_mapping:
            st.error("Selected value not in training data. Please retrain with full dataset.")
        else:

            input_data = pd.DataFrame({
                "brand": [brand_mapping[brand]],
                "year": [year],
                "fuel_type": [fuel_mapping[fuel_type]],
                "km_driven": [km_driven]
            })

            prediction = model.predict(input_data)[0]

            st.success(f"💰 Predicted Price: ₹ {prediction:,.2f}")