import os
import pickle
from datetime import datetime

import pandas as pd

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from app.database.database import get_session
from app.services.model import Prediction, StressLevelDatasetClean
from app import columns
from app.model_best_hyperparameters import model_best_hyperparameters


router = APIRouter()


@router.post("/train-model")
async def train_model(session: AsyncSession = Depends(get_session)):

    # 1. Отримання всіх записів із таблиці ознак
    result = await session.execute(
        select(StressLevelDatasetClean).order_by(
            StressLevelDatasetClean.ID
        )
    )

    records = result.scalars().all()

    # Перетворення записів у DataFrame
    df = pd.DataFrame([
        {
            column.name: getattr(record, column.name)
            for column in StressLevelDatasetClean.__table__.columns
        }
        for record in records
    ])

    
    df = df.dropna(subset=["stress_level"])

    
    train_df, new_input_df = train_test_split(
        df,
        train_size=0.9,
        random_state=42
    )

    train_df = train_df.copy()
    new_input_df = new_input_df.copy()

    mean_impute_columns = columns.X_columns

    mean_impute_values = {}

    for column in mean_impute_columns:
        mean_impute_values[column] = train_df[column].mean()

        train_df[column] = train_df[column].fillna(
            mean_impute_values[column]
        )

        new_input_df[column] = new_input_df[column].fillna(
            mean_impute_values[column]
        )

    scale_columns = [
        "depression",
        "anxiety_level"#,
        #"self_esteem"
    ]

    scale_values = {}

    for column in scale_columns:
        min_value = train_df[column].min()
        max_value = train_df[column].max()

        scale_values[column] = {
            "min": min_value,
            "max": max_value
        }

        if max_value != min_value:
            train_df[column] = (
                (train_df[column] - min_value) /
                (max_value - min_value)
            ) * 4 + 1

            new_input_df[column] = (
                (new_input_df[column] - min_value) /
                (max_value - min_value)
            ) * 4 + 1

    
    X_train = train_df[columns.X_columns]
    y_train = train_df[columns.y_column[0]]

    
    model = DecisionTreeClassifier(
        **model_best_hyperparameters.params
    )

    model.fit(X_train, y_train)

    # Створення папки для збереження моделі
    os.makedirs("models", exist_ok=True)

    # Збереження навченої моделі
    with open("models/finalized_model.sav", "wb") as file:
        pickle.dump(model, file)

    # Збереження параметрів попередньої обробки
    param_dict = {
        "mean_impute_values": mean_impute_values,
        "scale_values": scale_values
    }

    with open("models/param_dict.pickle", "wb") as file:
        pickle.dump(param_dict, file)

    # Отримання передбачень на навчальних даних
    y_pred = model.predict(X_train)

    #Запис передбачень у таблицю predictions
    predictions = []

    for p in y_pred:
        predictions.append(
            Prediction(
                prediction=int(p),
                source="train",
                timestamp=datetime.utcnow()
            )
        )

    session.add_all(predictions)

    # Збереження змін у базі даних
    await session.commit()

    return {
        "message": "Model trained and saved.",
        "train_records": len(train_df),
        "new_input_records": len(new_input_df)
    }