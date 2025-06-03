from sqlalchemy import Column, Integer, String, TIMESTAMP, Text
from sqlalchemy import JSON
from app.db.database import Base

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(TIMESTAMP, nullable=False)
    symbol = Column(String, nullable=False)
    timeframe = Column(String, nullable=False)
    signal = Column(String, nullable=False)
    cnn_lstm_probs = Column(JSON)
    xgb_probs = Column(JSON)
    hybrid_probs = Column(JSON)