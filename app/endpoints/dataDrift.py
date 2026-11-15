import json
import os
import tempfile

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset

from app.database.database import get_session
from app import columns


router = APIRouter()


@router.get("/monitor", response_class=HTMLResponse)
async def monitor(session: AsyncSession = Depends(get_session)):

    # Отримання даних з бази
    query = text("""
        SELECT i.input_data, i.source
        FROM dbo.inference_inputs AS i
        INNER JOIN dbo.predictions AS p ON p.id = i.id
        WHERE i.source IN ('train', 'inference')
          AND p.source = i.source
    """)

    result = await session.execute(query)
    rows = result.mappings().all()

    if not rows:
        raise HTTPException(status_code=404, detail="У базі немає даних. Спочатку виконай /train-model")

    df = pd.DataFrame([dict(r) for r in rows])

    # Розпакування JSON
    features = pd.json_normalize(
        df["input_data"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )
    )
    features = features[columns.X_columns].astype("float64")

    df = pd.concat([df[["source"]].reset_index(drop=True), features], axis=1)

    # Поділ на еталонні та нові дані
    reference = df[df["source"] == "train"].drop(columns="source").reset_index(drop=True)
    current = df[df["source"] == "inference"].drop(columns="source").reset_index(drop=True)

    if reference.empty or current.empty:
        raise HTTPException(status_code=400, detail="Потрібні і train, і inference дані")

    # Створення датасетів
    definition = DataDefinition(numerical_columns=columns.X_columns)

    reference_dataset = Dataset.from_pandas(reference, data_definition=definition)
    current_dataset = Dataset.from_pandas(current, data_definition=definition)

    # Створення Data Drift звіту
    report = Report(metrics=[
        DataDriftPreset(columns=columns.X_columns)
    ])

    evaluation = report.run(current_dataset, reference_dataset)

    # Повернення HTML-звіту (через тимчасовий файл)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "report.html")
        evaluation.save_html(path)
        with open(path, encoding="utf-8") as file:
            html = file.read()

    return HTMLResponse(content=html)