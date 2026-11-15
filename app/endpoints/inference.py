import pickle
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.database import get_session
from app.services.model import Prediction, InferenceInput, StressLevelDatasetClean
from app import columns


# ====== ПЕРЕМИКАЧ ======
# "holdout" -> прогноз на відкладених 10% датасету (запускати один раз)
# "json"    -> прогноз по одному запиту з JSON (додає по одному рядку)
MODE = "json"
# =======================


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


def load_artifacts():
    try:
        with open("models/finalized_model.sav", "rb") as file:
            model = pickle.load(file)
        with open("models/param_dict.pickle", "rb") as file:
            param_dict = pickle.load(file)
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Спочатку виконай /train-model")
    return model, param_dict


def preprocess(df: pd.DataFrame, param_dict: dict) -> pd.DataFrame:
    """Та сама обробка, що при навчанні."""
    df = df.copy()

    for column, mean_value in param_dict["mean_impute_values"].items():
        df[column] = df[column].fillna(mean_value)

    for column, values in param_dict["scale_values"].items():
        min_value, max_value = values["min"], values["max"]
        if max_value != min_value:
            df[column] = ((df[column] - min_value) / (max_value - min_value)) * 4 + 1

    return df[columns.X_columns].astype(float)


# ---------- режим 1: відкладені 10% ----------
async def predict_holdout(session: AsyncSession):
    model, param_dict = load_artifacts()

    result = await session.execute(
        select(StressLevelDatasetClean).order_by(StressLevelDatasetClean.ID)
    )
    records = result.scalars().all()

    df = pd.DataFrame([
        {c.name: getattr(r, c.name) for c in StressLevelDatasetClean.__table__.columns}
        for r in records
    ])
    df = df.dropna(subset=["stress_level"])

    # той самий split, що в /train-model
    _, holdout = train_test_split(df, train_size=0.9, random_state=42)
    raw = holdout.copy()                      # сирі значення для БД

    y_pred = model.predict(preprocess(holdout, param_dict))

    # прибираємо старі inference-рядки в обох таблицях, щоб не було дублів
    await session.execute(delete(Prediction).where(Prediction.source == "inference"))
    await session.execute(delete(InferenceInput).where(InferenceInput.source == "inference"))

    inputs, predictions = [], []
    for (_, row), p in zip(raw.iterrows(), y_pred):
        ts = row["timestamp"] if pd.notna(row["timestamp"]) else datetime.utcnow()
        features = {
            c: (None if pd.isna(row[c]) else int(row[c]))
            for c in columns.X_columns
        }
        inputs.append(InferenceInput(input_data=features, source="inference", timestamp=ts))
        predictions.append(
            Prediction(
                prediction=int(p),
                ground_truth=int(row["stress_level"]),
                source="inference",
                timestamp=ts,
            )
        )

    # спочатку входи, потім прогнози, в одному порядку -> id збігаються
    session.add_all(inputs)
    await session.flush()
    session.add_all(predictions)
    await session.commit()

    accuracy = float((y_pred == raw["stress_level"].values).mean())
    return {
        "message": "Holdout predictions saved.",
        "records": len(inputs),
        "accuracy_on_holdout": round(accuracy, 3),
    }


# ---------- режим 2: один запит із JSON ----------
async def predict_json(request: PredictRequest, session: AsyncSession):
    model, param_dict = load_artifacts()

    input_data = request.dict()
    X_input = preprocess(pd.DataFrame([input_data]), param_dict)

    missing = [c for c in columns.X_columns if X_input[c].isna().any()]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing values: {missing}")

    prediction = int(model.predict(X_input)[0])

    now = datetime.utcnow()

    pred_row = Prediction(
        prediction=prediction,
        ground_truth=None,               # мітка поки невідома
        source="inference",
        timestamp=now,
    )

    session.add(InferenceInput(input_data=input_data, source="inference", timestamp=now))
    session.add(pred_row)
    await session.flush()                # з'являється pred_row.id
    new_id = pred_row.id
    await session.commit()

    return {
        "message": "Prediction completed successfully.",
        "id": new_id,
        "input_data": input_data,
        "predicted_stress_level": prediction,
    }


# ---------- єдиний ендпоінт ----------
@router.post("/predict")
async def predict(
    request: Optional[PredictRequest] = None,
    session: AsyncSession = Depends(get_session),
):
    if MODE == "holdout":
        return await predict_holdout(session)

    if MODE == "json":
        return await predict_json(request or PredictRequest(), session)

    raise HTTPException(status_code=500, detail='MODE має бути "holdout" або "json"')



class GroundTruthRequest(BaseModel):
    id: int
    ground_truth: int = Field(ge=0, le=2)    # клас стресу: 0, 1 або 2


# @router.post("/ground-truth")
# async def set_ground_truth(
#     request: GroundTruthRequest,
#     session: AsyncSession = Depends(get_session),
# ):
#     result = await session.execute(
#         select(Prediction).where(
#             Prediction.id == request.id,
#             Prediction.source == "inference",
#         )
#     )
#     pred = result.scalar_one_or_none()

#     if pred is None:
#         raise HTTPException(status_code=404, detail="Запис з таким id не знайдено")

#     pred.ground_truth = request.ground_truth
#     await session.commit()

#     return {"message": "Ground truth saved.", "id": request.id}