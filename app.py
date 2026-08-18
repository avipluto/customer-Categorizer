import sys
import os

# This adds the parent folder (customer cat) so src/ is found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
from dotenv import load_dotenv
import os

import src.pipeline.prediction_pipeline
from src.pipeline.train_pipeline import TrainPipeline
from src.constant.application import *

import warnings
warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

# consistent variable name MONGO_DB_URL everywhere
print("MONGODB URL:", os.getenv("MONGO_DB_URL"))

app = FastAPI()

# Set up template directory
templates = Jinja2Templates(directory="templates")

# Enables CORS for all origins
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

from pydantic import BaseModel

class CustomerData(BaseModel):
    Age: int
    Education: int
    Marital_Status: int
    Parental_Status: int
    Children: int
    Income: int
    Total_Spending: int
    Days_as_Customer: int
    Recency: int
    Wines: int
    Fruits: int
    Meat: int
    Fish: float
    Sweets: int
    Gold: int
    Web: int
    Catalog: int
    Store: int
    Discount_Purchases: int
    Total_Promo: int
    NumWebVisitsMonth: int


# Train Model API
@app.get("/train")
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return JSONResponse(content={"message": "Training successful"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Test environment variables API
@app.get("/test_env")
async def test_env():
    mongo_url = os.getenv("MONGO_DB_URL")  
    return {"MONGODB URL": mongo_url}


# Render customer form UI
@app.get("/")
async def predictGetRouteClient(request: Request):
    try:
        return templates.TemplateResponse(
            "customer.html", {"request": request, "context": None}
        )
    except Exception as e:
        return JSONResponse(content={"status": False, "message": str(e)}, status_code=500)


# Predict API — HTML form submission
# NOTE: customer.html uses a plain <form method="post" action="/">, which browsers
# submit as application/x-www-form-urlencoded, NOT JSON. So this route now accepts
# individual Form(...) fields (matching each <input name="..."> in the form) instead
# of a JSON body, and returns the rendered template (with the predicted cluster
# filled into {{ context }}) instead of a raw JSON response.
@app.post("/")
async def predictRouteClient(
    request: Request,
    Age: int = Form(...),
    Education: int = Form(...),
    Marital_Status: int = Form(...),
    Parental_Status: int = Form(...),
    Children: int = Form(...),
    Income: int = Form(...),
    Total_Spending: int = Form(...),
    Days_as_Customer: int = Form(...),
    Recency: int = Form(...),
    Wines: int = Form(...),
    Fruits: int = Form(...),
    Meat: int = Form(...),
    Fish: float = Form(...),
    Sweets: int = Form(...),
    Gold: int = Form(...),
    Web: int = Form(...),
    Catalog: int = Form(...),
    Store: int = Form(...),
    Discount_Purchases: int = Form(...),
    Total_Promo: int = Form(...),
    NumWebVisitsMonth: int = Form(...),
):
    try:
        Data = CustomerData(
            Age=Age,
            Education=Education,
            Marital_Status=Marital_Status,
            Parental_Status=Parental_Status,
            Children=Children,
            Income=Income,
            Total_Spending=Total_Spending,
            Days_as_Customer=Days_as_Customer,
            Recency=Recency,
            Wines=Wines,
            Fruits=Fruits,
            Meat=Meat,
            Fish=Fish,
            Sweets=Sweets,
            Gold=Gold,
            Web=Web,
            Catalog=Catalog,
            Store=Store,
            Discount_Purchases=Discount_Purchases,
            Total_Promo=Total_Promo,
            NumWebVisitsMonth=NumWebVisitsMonth,
        )

        print("Received data:", Data.dict())

        input_data = [
            Data.Age,
            Data.Education,
            Data.Marital_Status,
            Data.Parental_Status,
            Data.Children,
            Data.Income,
            Data.Total_Spending,
            Data.Days_as_Customer,
            Data.Recency,
            Data.Wines,
            Data.Fruits,
            Data.Meat,
            Data.Fish,
            Data.Sweets,
            Data.Gold,
            Data.Web,
            Data.Catalog,
            Data.Store,
            Data.Discount_Purchases,
            Data.Total_Promo,
            Data.NumWebVisitsMonth
        ]

        prediction_pipeline = src.pipeline.prediction_pipeline.PredictionPipeline()
        predict_cluster = prediction_pipeline.predict_cluster(input_data)

        cluster_result = int(predict_cluster[0])

        return templates.TemplateResponse(
            "customer.html", {"request": request, "context": cluster_result}
        )

    except Exception as e:
        return templates.TemplateResponse(
            "customer.html", {"request": request, "context": None, "error": str(e)}
        )


# Run FastAPI application
if __name__ == "__main__":
    import uvicorn
    print("MONGODB URL:", os.getenv("MONGO_DB_URL"))  
    uvicorn.run(app, host="0.0.0.0", port=8080)