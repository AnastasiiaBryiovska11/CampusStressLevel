# Campus Stress Level — FastAPI ML API

API для навчання моделі класифікації рівня стресу та отримання передбачень на відкладеній частині датасету.

## Запуск сервера FastAPI

У терміналі відкрийте кореневу папку проєкту CampusStressLevel та виконайте:

### python -m app.main   

Після запуску сервер буде доступний за адресою:

1. API: http://127.0.0.1:8000
2. Інтерактивна документація Swagger: http://127.0.0.1:8000/docs 

Для зупинки сервера натисніть Ctrl+C у терміналі.

## Навчання моделі — POST /train-model

Ендпоінт /train-model читає записи з таблиці dbo.StressLevelDatasetClean, готує дані та навчає модель дерева рішень (DecisionTreeClassifier) на основі гіперпараметрів, що вказані в model_best_hyperparameters

Під час навчання:
1. Записи без цільового значення stress_level відкидаються.
2. Дані діляться на навчальну та відкладену вибірки у співвідношенні 90:10.
3. Пропущені значення ознак anxiety_level і depression заповнюються середніми значеннями навчальної вибірки.
4. Ознаки depression, anxiety_level і self_esteem масштабуються до діапазону [1; 5].
5. Модель навчається на 90% даних.
6. Навчена модель і параметри попередньої обробки зберігаються на диску.
7. Передбачення для навчальної вибірки записуються до dbo.predictions зі значенням source = 'train'.

Виклик через Swagger: відкрийте /docs, знайдіть POST /train-model, натисніть Try it out, а потім Execute.


Приклад:
	
```
Response body
{
  "message": "Model trained and saved.",
  "train_records": 990,
  "new_input_records": 111
}
```

Файли, створені після навчання:
1. finalized_model.sav — навчена модель.
2. param_dict.pickle — середні значення для заповнення пропусків і параметри масштабування.

## Передбачення — POST /predict

POST /predict отримує вхідні записи з бази даних. Ендпоінт повторно формує відкладені 10% записів із dbo.StressLevelDatasetClean, завантажує з диска модель та параметри попередньої обробки, виконує передбачення і зберігає результати.

Виклик через Swagger: відкрийте /docs, знайдіть POST /predict, натисніть Try it out, а потім Execute. Тіло запиту не потрібне.

Приклад:

```
Response body   
{
  "message": "Prediction completed successfully.",
  "inference_records": 111
}
```

Приклад структури JSON, що зберігається в inference_inputs
Наведений нижче JSON показує структуру одного запису. Значення ознак є демонстраційними. У поточному варіанті цей JSON не потрібно надсилати в /predict: ендпоінт формує його сам із даних БД.

```
{
  "ID": 123,
  "anxiety_level": 3,
  "self_esteem": 3,
  "mental_health_history": 0,
  "depression": 2,
  "headache": 2,
  "blood_pressure": 2,
  "sleep_quality": 3,
  "breathing_problem": 2,
  "noise_level": 2,
  "living_conditions": 3,
  "safety": 4,
  "basic_needs": 4,
  "academic_performance": 3,
  "study_load": 3,
  "teacher_student_relationship": 4,
  "future_career_concerns": 3,
  "social_support": 3,
  "peer_pressure": 2,
  "extracurricular_activities": 3,
  "bullying": 1,
  "stress_level": 0
}
```

stress_level у збереженому JSON — це фактичне значення з датасету для подальшого порівняння. Воно не передається моделі як вхідна ознака: модель отримує лише 20 ознак із columns.X_columns.

## Структура бази даних

Проєкт використовує SQL Server. Основні таблиці розташовані у схемі dbo.

dbo.StressLevelDatasetClean
Основна таблиця з підготовленим датасетом. Містить 20 вхідних ознак, цільову змінну stress_level, а також службові поля ID і timestamp.

Вхідні ознаки:
1. anxiety_level
2. self_esteem
3. mental_health_history
4. depression
5. headache
6. blood_pressure
7. sleep_quality
8. breathing_problem
9. noise_level
10. living_conditions
11. safety
12. basic_needs
13. academic_performance
14. study_load
15. teacher_student_relationship
16. future_career_concerns
17. social_support
18. peer_pressure
19. extracurricular_activities
20. bullying

Цільова змінна: stress_level.

dbo.predictions

Зберігає результати роботи моделі:

1. ID - Ідентифікатор запису
2. prediction - Передбачений рівень стресу
3. timestamp - Дата й час створення запису
4. source - Джерело передбачення: train або inference


dbo.inference_inputs

Зберігає вхідні параметри, для яких модель виконала передбачення:

ID - Ідентифікатор запису
input_data - Вхідні параметри у форматі JSON
timestamp - Дата й час створення запису

## Приклади збережених передбачень

Переглянути передбачення для навчальної та відкладеної вибірок можна SQL-запитами:

``` 
 WITH predictions AS (
    SELECT
        ROW_NUMBER() OVER (ORDER BY id) AS row_num,
        prediction
    FROM dbo.predictions
    WHERE source = 'inference'
),

inputs AS (
    SELECT
        ROW_NUMBER() OVER (ORDER BY id) AS row_num,

        TRY_CONVERT(
            INT,
            JSON_VALUE(input_data, '$.ID')
        ) AS dataset_id,

        TRY_CONVERT(
            INT,
            JSON_VALUE(input_data, '$.stress_level')
        ) AS actual_stress_level

    FROM dbo.inference_inputs
)

SELECT
    inputs.dataset_id,
    inputs.actual_stress_level,
    predictions.prediction,

    CASE
        WHEN inputs.actual_stress_level = predictions.prediction
        THEN 'Правильно'
        ELSE 'Неправильно'
    END AS result

FROM inputs
JOIN predictions
    ON inputs.row_num = predictions.row_num

ORDER BY inputs.row_num;
```


За результатами одного тестового запуску модель правильно класифікувала 94 із 111 записів. Точність (Accuracy) становила 84,68%. Це результат конкретного запуску на відкладеній вибірці.

Приклад:
1. dataset_id = 413
2. actual_stress_level = 1
3. prediction = 1
4. result = "Правильно"