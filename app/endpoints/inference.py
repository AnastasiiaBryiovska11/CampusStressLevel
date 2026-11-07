
import pickle
import pandas as pd

from typing import Optional
from pydantic import BaseModel

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.database import get_session
from app.services.model import Prediction, InferenceInput
from app import columns


router = APIRouter()


class PredictRequest(BaseModel):
    anxiety_level: Optional[int] = 4
    self_esteem: Optional[int] = 3
    mental_health_history: Optional[int] = 1
    depression: Optional[int] = 3
    headache: Optional[int] = 4
    blood_pressure: Optional[int] = 3
    sleep_quality: Optional[int] = 1
    breathing_problem: Optional[int] = 3
    noise_level: Optional[int] = 4
    living_conditions: Optional[int] = 2
    safety: Optional[int] = 2
    basic_needs: Optional[int] = 2
    academic_performance: Optional[int] = 2
    study_load: Optional[int] = 3
    teacher_student_relationship: Optional[int] = 1
    future_career_concerns: Optional[int] = 4
    social_support: Optional[int] = 1
    peer_pressure: Optional[int] = 4
    extracurricular_activities: Optional[int] = 4
    bullying: Optional[int] = 5


@router.post("/predict")
async def predict(
    request: PredictRequest,
    session: AsyncSession = Depends(get_session)
):

    input_data = request.dict()

    input_df = pd.DataFrame([input_data])

   
    with open("models/finalized_model.sav", "rb") as file:
        model = pickle.load(file)


    with open("models/param_dict.pickle", "rb") as file:
        param_dict = pickle.load(file)

    
    for column, mean_value in param_dict["mean_impute_values"].items():
        input_df[column] = input_df[column].fillna(mean_value)

    
    missing = [
        column for column in columns.X_columns
        if input_df[column].isna().any()
    ]

    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing values: {missing}"
        )

   
    for column, values in param_dict["scale_values"].items():
        min_value = values["min"]
        max_value = values["max"]

        if max_value != min_value:
            input_df[column] = (
                (input_df[column] - min_value) /
                (max_value - min_value)
            ) * 4 + 1

  
    X_input = input_df[columns.X_columns]

    X_input = input_df[columns.X_columns]

    print("Дані перед predict:")
    print(X_input.to_string())

    prediction = model.predict(X_input)

   
    prediction = model.predict(X_input)

    print("Вхідні дані:")
    print(X_input)

    print("Прогноз:")
    print(prediction)

    print("Класи, які знає модель:")
    print(model.classes_)
    
    session.add(
        Prediction(
            prediction=int(prediction[0]),
            source="inference"
        )
    )

    print("Дані після обробки:")
    print(X_input if "X_input" in locals() else input_df[columns.X_columns].to_string())

    
    session.add(
        InferenceInput(
            input_data=input_data
        )
    )

    await session.commit()

    
    return {
        "message": "Prediction completed successfully.",
        "input_data": input_data,
        "predicted_stress_level": int(prediction[0])
    }
