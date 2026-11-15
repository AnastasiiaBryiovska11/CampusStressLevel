import os
import pickle
from datetime import datetime

import pandas as pd
from fastapi import APIRouter, Depends
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.model import Prediction, InferenceInput, StressLevelDatasetClean
from app.database.database import get_session
from app import columns
from app.model_best_hyperparameters import model_best_hyperparameters


router = APIRouter()


def build_rows(raw_df, preds, source):
    """Для кожного рядка створює вхід (InferenceInput) і прогноз (Prediction)."""
    inputs, predictions = [], []

    for (_, row), p in zip(raw_df.iterrows(), preds):
        ts = row["timestamp"] if pd.notna(row["timestamp"]) else datetime.utcnow()

        features = {
            c: (None if pd.isna(row[c]) else int(row[c]))
            for c in columns.X_columns
        }

        inputs.append(
            InferenceInput(
                input_data=features,
                source=source,
                timestamp=ts,
            )
        )
        predictions.append(
            Prediction(
                prediction=int(p),
                ground_truth=int(row["stress_level"]),
                source=source,
                timestamp=ts,
            )
        )

    return inputs, predictions


@router.post("/train-model")
async def train_model(session: AsyncSession = Depends(get_session)):

    # 1. Отримання всіх записів із таблиці ознак
    result = await session.execute(
        select(StressLevelDatasetClean).order_by(StressLevelDatasetClean.ID)
    )
    records = result.scalars().all()

    df = pd.DataFrame([
        {
            column.name: getattr(record, column.name)
            for column in StressLevelDatasetClean.__table__.columns
        }
        for record in records
    ])

    df = df.dropna(subset=["stress_level"])

    # 2. 90% на навчання, 10% відкладаємо (їх використає /predict)
    train_df, new_input_df = train_test_split(
        df,
        train_size=0.9,
        random_state=42
    )

    train_df = train_df.copy()
    new_input_df = new_input_df.copy()

    train_raw = train_df.copy()          # сирі значення для запису в БД

    # 3. Заповнення пропусків
    mean_impute_values = {}

    for column in columns.X_columns:
        mean_impute_values[column] = train_df[column].mean()

        train_df[column] = train_df[column].fillna(mean_impute_values[column])
        new_input_df[column] = new_input_df[column].fillna(mean_impute_values[column])

    # 4. Масштабування
    scale_columns = [
        "depression",
        "anxiety_level",
    ]

    scale_values = {}

    for column in scale_columns:
        min_value = train_df[column].min()
        max_value = train_df[column].max()

        scale_values[column] = {"min": min_value, "max": max_value}

        if max_value != min_value:
            train_df[column] = (
                (train_df[column] - min_value) / (max_value - min_value)
            ) * 4 + 1

            new_input_df[column] = (
                (new_input_df[column] - min_value) / (max_value - min_value)
            ) * 4 + 1

    # 5. Навчання
    X_train = train_df[columns.X_columns]
    y_train = train_df[columns.y_column[0]]

    model = DecisionTreeClassifier(**model_best_hyperparameters.params)
    model.fit(X_train, y_train)

    # 6. Збереження моделі та параметрів обробки
    os.makedirs("models", exist_ok=True)

    with open("models/finalized_model.sav", "wb") as file:
        pickle.dump(model, file)

    param_dict = {
        "mean_impute_values": mean_impute_values,
        "scale_values": scale_values,
    }

    with open("models/param_dict.pickle", "wb") as file:
        pickle.dump(param_dict, file)

    # 7. Прогнози на навчальних даних
    y_pred = model.predict(X_train)

    # 8. Повне очищення таблиць (щоб id збігались)
    await session.execute(delete(Prediction))
    await session.execute(delete(InferenceInput))

    # 9. Запис лише train-рядків
    train_inputs, train_preds = build_rows(train_raw, y_pred, "train")

    # спочатку входи, потім прогнози, в одному порядку -> id збігаються
    session.add_all(train_inputs)
    await session.flush()
    session.add_all(train_preds)

    await session.commit()

    return {
        "message": "Model trained and saved.",
        "train_records": len(train_df),
        "holdout_records": len(new_input_df),
    }