from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from predict import predict_house_price


# Create FastAPI app
app = FastAPI(
    title="House Price Prediction",
    description="Machine Learning House Price Prediction",
    version="1.0"
)


# Templates folder
templates = Jinja2Templates(
    directory="templates"
)


# Static files
app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# Home page
@app.get("/", response_class=HTMLResponse)
def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"error": None}
    )


# Prediction
@app.post("/predict", response_class=HTMLResponse)
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

    try:

        # Send input to ML model
        result = predict_house_price(
            Area,
            Bedrooms,
            Bathrooms,
            Floors,
            YearBuilt,
            Location,
            Condition,
            Garage
        )

        # Show prediction result
        return templates.TemplateResponse(
            request=request,
            name="result.html",
            context={
                "prediction": result["prediction"],
                "best_model": result["best_model"],
                "predictions": result["predictions"],
                "input_data": result["input_data"],
                "error": None
            }
        )

    except Exception as e:

        # Show error on result page
        return templates.TemplateResponse(
            request=request,
            name="result.html",
            context={
                "prediction": None,
                "best_model": None,
                "predictions": [],
                "input_data": {
                    "Area": Area,
                    "Bedrooms": Bedrooms,
                    "Bathrooms": Bathrooms,
                    "Floors": Floors,
                    "YearBuilt": YearBuilt,
                    "Location": Location,
                    "Condition": Condition,
                    "Garage": Garage
                },
                "error": str(e)
            }
        )