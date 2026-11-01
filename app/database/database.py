from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine import URL

DATABASE_URL = URL.create(
    "mssql+aioodbc",
    host="Oleg",
    database="StudentStressMonitoring",
    query={
        "driver": "ODBC Driver 17 for SQL Server",
        "trusted_connection": "yes",
        "TrustServerCertificate": "yes"
    }
)

engine = create_async_engine(DATABASE_URL, echo=True)

AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session





# import asyncio
# from sqlalchemy import text

# async def test_connection():
#     try:
#         async with engine.connect() as connection:
#             result = await connection.execute(text("SELECT DB_NAME()"))
#             database_name = result.scalar()

#             print("Підключення успішне!")
#             print("База даних:", database_name)

#     except Exception as e:
#         print("Помилка підключення:", e)

# asyncio.run(test_connection())