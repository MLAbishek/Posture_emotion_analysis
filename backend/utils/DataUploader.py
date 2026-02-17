from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, Boolean, DateTime, ForeignKey, create_engine
from sqlalchemy.orm import declarative_base, relationship, Session
from db_config import SessionLocal, engine

Base = declarative_base()


class SessionModel(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True, index=True)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    
    

class LogModel(Base):
    __tablename__ = "logdetails"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    neck_angle = Column(Float)
    is_slouching = Column(Boolean)
    ear_value = Column(Float)
    is_drowsy = Column(Boolean)
    emotion = Column(String)
    emotion_confidence = Column(Float)
    


class DataUploader:
    def __init__(self):
        self.db: Session = SessionLocal()
        self.current_session_id = None
        self.frame_counter = 0
        self.log_interval = 30 # Log every 30 frames (approx 1 sec)
        print("Connected to Database via DataUploader.")

    def start_session(self):

        try:
            new_session = SessionModel(start_time=datetime.now())
            self.db.add(new_session)
            self.db.commit()
            self.db.refresh(new_session)
            self.current_session_id = new_session.id
            print(f"Session Started: ID {self.current_session_id}")
            return self.current_session_id
        except Exception as e:
            print(f"Error starting session: {e}")
            self.db.rollback()
            return None

    def log_data(self, neck_angle, is_slouching, ear_value, is_drowsy, emotion, emotion_confidence=0.0):

        if not self.current_session_id:
            return

        self.frame_counter += 1
        if self.frame_counter % self.log_interval != 0:
            return

        try:
            new_log = LogModel(
                session_id=self.current_session_id,
                timestamp=datetime.now(),
                neck_angle=float(neck_angle),
                is_slouching=bool(is_slouching),
                ear_value=float(ear_value),   
                is_drowsy=bool(is_drowsy),
                emotion=str(emotion),
                emotion_confidence=float(emotion_confidence) 
            )
            self.db.add(new_log)
            self.db.commit()
        except Exception as e:
            print(f"Error logging data: {e}")
            self.db.rollback()

    def end_session(self):
        
        if not self.current_session_id:
            return

        try:
            session = self.db.query(SessionModel).filter(SessionModel.id == self.current_session_id).first()
            if session:
                session.end_time = datetime.now()
                self.db.commit()
                print(f"Session Ended: ID {self.current_session_id}")
        except Exception as e:
            print(f"Error ending session: {e}")
            self.db.rollback()
        finally:
            self.db.close()
