
from sqlalchemy import Column, String, Integer, DateTime
from sqlalchemy import JSON
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    prediction = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)
    source = Column(String)


class InferenceInput(Base):
    __tablename__ = "inference_inputs"

    id = Column(Integer, primary_key=True, index=True)
    input_data = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)


class StressLevelDatasetClean(Base):
    __tablename__ = "StressLevelDatasetClean"
    __table_args__ = {"schema": "dbo"}

    anxiety_level = Column(Integer)
    self_esteem = Column(Integer)
    mental_health_history = Column(Integer)
    depression = Column(Integer)
    headache = Column(Integer)
    blood_pressure = Column(Integer)
    sleep_quality = Column(Integer)
    breathing_problem = Column(Integer)
    noise_level = Column(Integer)
    living_conditions = Column(Integer)
    safety = Column(Integer)
    basic_needs = Column(Integer)
    academic_performance = Column(Integer)
    study_load = Column(Integer)
    teacher_student_relationship = Column(Integer)
    future_career_concerns = Column(Integer)
    social_support = Column(Integer)
    peer_pressure = Column(Integer)
    extracurricular_activities = Column(Integer)
    bullying = Column(Integer)
    stress_level = Column(Integer)
    timestamp = Column(DateTime)
    ID = Column(Integer, primary_key=True)