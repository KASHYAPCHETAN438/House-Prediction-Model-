# ==========================================
# HOUSE PRICE PREDICTION - ML
# ==========================================

from pathlib import Path
import pandas as pd
import joblib


# 1. Load model bundle
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "house_price_models.joblib"

model_bundle = joblib.load(MODEL_PATH)

models = model_bundle["models"]
metrics_df = model_bundle["metrics"]


# 2. Prepare input according to model features
def prepare_input(input_data, model):

    # Convert categorical columns into numerical columns
    data = pd.get_dummies(
        input_data,
        columns=["Location", "Condition", "Garage"],
        dtype=int
    )

    # Match the features used during training
    if hasattr(model, "feature_names_in_"):
        data = data.reindex(
            columns=model.feature_names_in_,
            fill_value=0
        )

    return data


# 3. Get model performance
def get_metrics(model_name):

    result = metrics_df[
        metrics_df["Model"].astype(str) == str(model_name)
    ]

    if result.empty:
        return 0.0, 0.0, 0.0

    return (
        float(result["R2 Score"].iloc[0]),
        float(result["MAE"].iloc[0]),
        float(result["RMSE"].iloc[0])
    )


# 4. Main prediction function
def predict_house_price(
    Area,
    Bedrooms,
    Bathrooms,
    Floors,
    YearBuilt,
    Location,
    Condition,
    Garage
):

    # Create input DataFrame
    input_data = pd.DataFrame({
        "Area": [Area],
        "Bedrooms": [Bedrooms],
        "Bathrooms": [Bathrooms],
        "Floors": [Floors],
        "YearBuilt": [YearBuilt],
        "Location": [Location],
        "Condition": [Condition],
        "Garage": [Garage]
    })


    # Store predictions
    predictions = []


    # Predict using every model
    for model_name, model in models.items():

        try:
            model_input = prepare_input(
                input_data.copy(),
                model
            )

            price = float(
                model.predict(model_input)[0]
            )

            r2, mae, rmse = get_metrics(model_name)

            predictions.append({
                "model": model_name,
                "prediction": round(price, 2),
                "r2": round(r2, 4),
                "mae": round(mae, 2),
                "rmse": round(rmse, 2)
            })

        except Exception:
            continue


    # Make sure at least one model worked
    if not predictions:
        raise ValueError(
            "No model was able to make a prediction."
        )


    # Select model having highest R²
    best_result = max(
        predictions,
        key=lambda x: x["r2"]
    )


    # Return result to FastAPI
    return {
        "prediction": best_result["prediction"],
        "best_model": best_result["model"],
        "predictions": predictions,

        "input_data": {
            "Area": Area,
            "Bedrooms": Bedrooms,
            "Bathrooms": Bathrooms,
            "Floors": Floors,
            "YearBuilt": YearBuilt,
            "Location": Location,
            "Condition": Condition,
            "Garage": Garage
        }
    }