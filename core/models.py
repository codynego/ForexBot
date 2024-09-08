from sqlalchemy import Column, Integer, String, Text, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class BaseModel(Base):
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    time = Column(DateTime, default=datetime.utcnow)

class Trade(BaseModel):
    __tablename__ = "Trade"
    


class Signal(BaseModel):
    __tablename__ = "signal"
    

class Market(BaseModel):
    __tablename__ = "market"
    