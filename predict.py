# ==========================================
# HOUSE PRICE PREDICTION - ML
# ==========================================

from pathlib import Path

import pandas as pd

import joblib


# ==========================================
# LOAD MODEL BUNDLE
# ==========================================

BASE_DIR = Path(
    __file__
).resolve().parent


MODEL_PATH = (
    BASE_DIR /
    "house_price_models.joblib"
)


# Load saved model bundle

model_bundle = joblib.load(
    MODEL_PATH
)


# Extract models

models = model_bundle["models"]


# Extract metrics

metrics_df = model_bundle["metrics"]


# ==========================================
# PREPARE INPUT
# ==========================================

def prepare_input(
    input_data,
    model
):

    """
    Prepare user input according to
    the features used during model training.
    """


    # --------------------------------------
    # Convert categorical features
    # into numerical dummy variables
    # --------------------------------------

    data = pd.get_dummies(

        input_data,

        columns=[
            "Location",
            "Condition",
            "Garage"
        ],

        dtype=int

    )


    # --------------------------------------
    # Match training features
    # --------------------------------------

    if hasattr(
        model,
        "feature_names_in_"
    ):

        data = data.reindex(

            columns=model.feature_names_in_,

            fill_value=0

        )


    return data


# ==========================================
# GET MODEL METRICS
# ==========================================

def get_metrics(
    model_name
):

    """
    Get R², MAE and RMSE
    for a particular model.
    """


    result = metrics_df[

        metrics_df["Model"]
        .astype(str)
        ==
        str(model_name)

    ]


    # --------------------------------------
    # If metrics not found
    # --------------------------------------

    if result.empty:

        return (
            0.0,
            0.0,
            0.0
        )


    # --------------------------------------
    # Return metrics
    # --------------------------------------

    return (

        float(
            result["R2 Score"].iloc[0]
        ),

        float(
            result["MAE"].iloc[0]
        ),

        float(
            result["RMSE"].iloc[0]
        )

    )


# ==========================================
# MAIN PREDICTION FUNCTION
# ==========================================

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

    """
    Generate house price predictions
    using every saved ML model.
    """


    # ======================================
    # CREATE INPUT DATAFRAME
    # ======================================

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


    # ======================================
    # STORE ALL PREDICTIONS
    # ======================================

    predictions = []


    # ======================================
    # RUN EVERY MODEL
    # ======================================

    for model_name, model in models.items():

        try:

            # --------------------------------
            # Prepare input
            # --------------------------------

            model_input = prepare_input(

                input_data.copy(),

                model

            )


            # --------------------------------
            # Prediction
            # --------------------------------

            price = float(

                model.predict(
                    model_input
                )[0]

            )


            # --------------------------------
            # Get metrics
            # --------------------------------

            r2, mae, rmse = get_metrics(

                model_name

            )


            # --------------------------------
            # Store result
            # --------------------------------

            predictions.append({

                "model":
                    str(model_name),

                "prediction":
                    round(price, 2),

                "r2":
                    round(r2, 4),

                "mae":
                    round(mae, 2),

                "rmse":
                    round(rmse, 2)

            })


            # --------------------------------
            # Terminal output
            # --------------------------------

            print(
                f"{model_name}: "
                f"₹{price:,.2f}"
            )


        except Exception as e:

            # --------------------------------
            # Don't stop other models
            # --------------------------------

            print(
                f"Error in {model_name}: "
                f"{e}"
            )

            continue


    # ======================================
    # CHECK PREDICTIONS
    # ======================================

    if not predictions:

        raise ValueError(

            "No model was able to "
            "make a prediction."

        )


    # ======================================
    # FIND BEST MODEL
    # ======================================

    best_result = max(

        predictions,

        key=lambda x: x["r2"]

    )


    # ======================================
    # FINAL RESULT
    # ======================================

    return {

        # Best model prediction

        "prediction":
            best_result["prediction"],


        # Best model name

        "best_model":
            best_result["model"],


        # ALL model predictions

        "predictions":
            predictions,


        # User input

        "input_data": {

            "Area":
                Area,

            "Bedrooms":
                Bedrooms,

            "Bathrooms":
                Bathrooms,

            "Floors":
                Floors,

            "YearBuilt":
                YearBuilt,

            "Location":
                Location,

            "Condition":
                Condition,

            "Garage":
                Garage

        }

    }