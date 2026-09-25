from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from predict import predict_house_price


# ==========================================
# FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="House Price Prediction",
    description="Machine Learning House Price Prediction",
    version="1.0"
)


# ==========================================
# TEMPLATES
# ==========================================

templates = Jinja2Templates(
    directory="templates"
)


# ==========================================
# STATIC FILES
# ==========================================

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# ==========================================
# HOME PAGE
# ==========================================

@app.get(
    "/",
    response_class=HTMLResponse
)
def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "error": None
        }
    )


# ==========================================
# PREDICTION
# ==========================================

@app.post(
    "/predict",
    response_class=HTMLResponse
)
def predict(

    request: Request,

    Area: float = Form(...),

    Bedrooms: int = Form(...),

    Bathrooms: int = Form(...),

    Floors: int = Form(...),

    YearBuilt: int = Form(...),

    Location: str = Form(...),

    Condition: str = Form(...),

    Garage: str = Form(...)

):

    # ======================================
    # INPUT DATA
    # ======================================

    input_data = {

        "Area": Area,

        "Bedrooms": Bedrooms,

        "Bathrooms": Bathrooms,

        "Floors": Floors,

        "YearBuilt": YearBuilt,

        "Location": Location,

        "Condition": Condition,

        "Garage": Garage

    }


    try:

        # ==================================
        # CALL ML PREDICTION
        # ==================================

        result = predict_house_price(

            Area=Area,

            Bedrooms=Bedrooms,

            Bathrooms=Bathrooms,

            Floors=Floors,

            YearBuilt=YearBuilt,

            Location=Location,

            Condition=Condition,

            Garage=Garage

        )


        # ==================================
        # DEBUG INFORMATION
        # ==================================

        print("\n====================================")
        print("HOUSE PRICE PREDICTION")
        print("====================================")

        print("Input:")
        print(input_data)

        print(
            "Total Models:",
            len(result["predictions"])
        )

        print("\nModel Predictions:")

        for item in result["predictions"]:

            print(
                f"{item['model']} -> "
                f"₹{item['prediction']:,.2f}"
            )

        print("\nBest Model:")
        print(result["best_model"])

        print("====================================\n")


        # ==================================
        # RESULT PAGE
        # ==================================

        return templates.TemplateResponse(

            request=request,

            name="result.html",

            context={

                "prediction":
                    result["prediction"],

                "best_model":
                    result["best_model"],

                "predictions":
                    result["predictions"],

                "input_data":
                    result["input_data"],

                "error":
                    None

            }

        )


    except Exception as e:

        # ==================================
        # ERROR
        # ==================================

        print("\nPrediction Error:")
        print(str(e))


        return templates.TemplateResponse(

            request=request,

            name="result.html",

            context={

                "prediction":
                    None,

                "best_model":
                    None,

                "predictions":
                    [],

                "input_data":
                    input_data,

                "error":
                    str(e)

            }

        )