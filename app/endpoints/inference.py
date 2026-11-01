import pickle
import json
import pandas as pd

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sklearn.model_selection import train_test_split

from app.database.database import get_session
from app.services.model import Prediction, InferenceInput, StressLevelDatasetClean
from app import columns


router = APIRouter()


@router.post("/predict")
async def predict(
    session: AsyncSession = Depends(get_session)
):

    # 1. Отримання вхідних даних із БД
    result = await session.execute(
        select(StressLevelDatasetClean).order_by(
            StressLevelDatasetClean.ID
        )
    )

    records = result.scalars().all()

    input_df = pd.DataFrame([
        {
            column.name: getattr(record, column.name)
            for column in StressLevelDatasetClean.__table__.columns
        }
        for record in records
    ])

    input_df = input_df.dropna(subset=["stress_level"])

    # 2. Отримання відкладених 10% даних
    _, input_df = train_test_split(
        input_df,
        train_size=0.9,
        random_state=42
    )

    input_df = input_df.copy()

    # Збереження початкових вхідних параметрів
    input_data = json.loads(
        input_df[
            ["ID"] + columns.X_columns + ["stress_level"]
        ].to_json(orient="records")
    )

    # 3. Завантаження моделі з диска
    with open("models/finalized_model.sav", "rb") as file:
        model = pickle.load(file)

    # 4. Завантаження параметрів попередньої обробки
    with open("models/param_dict.pickle", "rb") as file:
        param_dict = pickle.load(file)

    # 5. Заповнення пропущених значень
    for column, mean_value in param_dict["mean_impute_values"].items():
        input_df[column] = input_df[column].fillna(mean_value)

    # Перевірка, чи всі ознаки мають значення
    missing = [
        column for column in columns.X_columns
        if input_df[column].isna().any()
    ]

    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing values: {missing}"
        )

    # 6. Масштабування ознак
    for column, values in param_dict["scale_values"].items():
        min_value = values["min"]
        max_value = values["max"]

        if max_value != min_value:
            input_df[column] = (
                (input_df[column] - min_value) /
                (max_value - min_value)
            ) * 4 + 1

    # 7. Виконання передбачення
    X_input = input_df[columns.X_columns]
    prediction = model.predict(X_input)

    # 8. Збереження передбачення та вхідних параметрів
    session.add_all([
        Prediction(
            prediction=int(p),
            source="inference"
        )
        for p in prediction
    ] + [
        InferenceInput(
            input_data=row
        )
        for row in input_data
    ])

    await session.commit()

    # 9. Повернення результату
    return {
        "message": "Prediction completed successfully.",
        "inference_records": len(prediction)
    }