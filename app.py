import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, field_validator


model = joblib.load("diabetes_model.pkl")
training_columns = joblib.load("training_columns.pkl")

app = FastAPI(title="Diabetes Prediction API")


class PatientData(BaseModel):
    age: float
    urea: float
    cr: float
    hba1c: float
    chol: float
    tg: float
    hdl: float
    ldl: float
    vldl: float
    bmi: float
    gender: str

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, value):
        value = value.strip().upper()
        if value not in ["M", "F"]:
            raise ValueError("Gender must be either 'M' or 'F'")
        return value


@app.get("/")
def health_check():
    return {"status": "API is running"}


@app.post("/predict")
def predict_diabetes(data: PatientData):
    input_dict = data.model_dump()

    input_df = pd.DataFrame([input_dict])

    input_df["gender"] = input_df["gender"].str.upper()

    input_df = pd.get_dummies(input_df, columns=["gender"], drop_first=False)

    input_df = input_df.reindex(columns=training_columns, fill_value=0)

    prediction = model.predict(input_df)[0]

    return {
        "prediction": prediction
    }