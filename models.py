# models.py
from pydantic import BaseModel
from typing import Optional, List, Dict

class UploadResponse(BaseModel):
    columns: list
    date_col: str
    value_col: str
    freq: str
    data_points: int
    time_range: list
    mean_value: float
    min_value: float
    max_value: float
    std_value: float
    range_value: float