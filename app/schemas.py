from pydantic import BaseModel
from typing import Optional


class PredictRequest(BaseModel):
    anxiety_level: Optional[int] = None
    self_esteem: Optional[int] = None
    mental_health_history: Optional[int] = None
    depression: Optional[int] = None
    headache: Optional[int] = None
    blood_pressure: Optional[int] = None
    sleep_quality: Optional[int] = None
    breathing_problem: Optional[int] = None
    noise_level: Optional[int] = None
    living_conditions: Optional[int] = None
    safety: Optional[int] = None
    basic_needs: Optional[int] = None
    academic_performance: Optional[int] = None
    study_load: Optional[int] = None
    teacher_student_relationship: Optional[int] = None
    future_career_concerns: Optional[int] = None
    social_support: Optional[int] = None
    peer_pressure: Optional[int] = None
    extracurricular_activities: Optional[int] = None
    bullying: Optional[int] = None