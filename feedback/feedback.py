from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

from .session_store import get_last_ai_response, clear_session

load_dotenv()

feedback_router = APIRouter()

# Database connection
db_url = os.getenv('db_url', 'postgresql+psycopg2://postgres:root@localhost:5432/call_agent')
engine = create_engine(db_url)


class FeedbackValueRequest(BaseModel):
    feedback_value: int = Field(..., ge=0, le=10, description="Feedback rating from 0 to 10")


class FeedbackMessageRequest(BaseModel):
    feedback_message: str


class FeedbackFullRequest(BaseModel):
    """Combined feedback with optional session_id to link ai_response"""
    feedback_value: int = Field(..., ge=0, le=10, description="Feedback rating from 0 to 10")
    feedback_message: str = ""
    session_id: str | None = None


class FeedbackResponse(BaseModel):
    status: str
    message: str
    feedback_id: int | None = None


def save_ai_response_to_db(ai_response: str, session_id: str | None = None) -> int | None:
    """Save ai_response to customer_feedback table (call_agent DB). Returns feedback_id."""
    ensure_feedback_table()
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    INSERT INTO customer_feedback (ai_response)
                    VALUES (:ai_response)
                    RETURNING feedback_id;
                """),
                {"ai_response": ai_response}
            )
            conn.commit()
            return result.fetchone()[0]
    except Exception as e:
        print(f"Error saving ai_response: {e}")
        return None


def ensure_feedback_table():
    """Create feedback table with feedback_value, feedback_message, and ai_response columns if it doesn't exist"""
    try:
        with engine.connect() as conn:
            create_table_query = """
            CREATE TABLE IF NOT EXISTS customer_feedback (
                feedback_id SERIAL PRIMARY KEY,
                customer_nic VARCHAR(50),
                connection_number VARCHAR(50),
                feedback_value INTEGER CHECK (feedback_value >= 0 AND feedback_value <= 10),
                feedback_message TEXT,
                ai_response TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            conn.execute(text(create_table_query))
            conn.commit()
            # Add ai_response column if table already existed without it (PostgreSQL)
            try:
                check_col = conn.execute(text("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name='customer_feedback' AND column_name='ai_response';
                """)).fetchone()
                if not check_col:
                    conn.execute(text("ALTER TABLE customer_feedback ADD COLUMN ai_response TEXT;"))
                    conn.commit()
            except Exception:
                pass
    except Exception as e:
        print(f"Error creating table: {str(e)}")


@feedback_router.post("/feedback-value", response_model=FeedbackResponse)
async def submit_feedback_value(request: FeedbackValueRequest):
    """
    Endpoint 1: Submit customer feedback rating (0-10)
    Saves feedback_value to database with customer NIC and connection details
    """
    ensure_feedback_table()
    
    try:
        with engine.connect() as conn:
            # Insert feedback value record
            insert_query = """
            INSERT INTO customer_feedback (feedback_value)
            VALUES (:feedback)
            RETURNING feedback_id;
            """
            result = conn.execute(
                text(insert_query),
                {"feedback": request.feedback_value}
            )
            conn.commit()
            
            feedback_id = result.fetchone()[0]
            
            return FeedbackResponse(
                status="success",
                message=f"Feedback value ({request.feedback_value}/10) saved successfully",
                feedback_id=feedback_id
            )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error saving feedback value: {str(e)}"
        )


@feedback_router.post("/feedback-full", response_model=FeedbackResponse)
async def submit_feedback_full(request: FeedbackFullRequest):
    """
    Combined endpoint: Submit feedback_value, feedback_message, and ai_response (from session_id).
    Saves to customer_feedback with ai_response column.
    """
    ensure_feedback_table()
    ai_response = ""
    if request.session_id:
        ai_response = get_last_ai_response(request.session_id)
        clear_session(request.session_id)
    try:
        with engine.connect() as conn:
            insert_query = """
            INSERT INTO customer_feedback (feedback_value, feedback_message, ai_response)
            VALUES (:feedback_value, :feedback_message, :ai_response)
            RETURNING feedback_id;
            """
            result = conn.execute(
                text(insert_query),
                {
                    "feedback_value": request.feedback_value,
                    "feedback_message": request.feedback_message,
                    "ai_response": ai_response or None,
                }
            )
            conn.commit()
            feedback_id = result.fetchone()[0]
            return FeedbackResponse(
                status="success",
                message=f"Feedback ({request.feedback_value}/10) සහ ai_response සංරක්ෂණය කරන ලදී",
                feedback_id=feedback_id
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving feedback: {str(e)}")


@feedback_router.post("/feedback-message", response_model=FeedbackResponse)
async def submit_feedback_message(request: FeedbackMessageRequest):
    """
    Endpoint 2: Submit customer feedback message
    Saves feedback_message text to database with customer NIC and connection details
    """
    ensure_feedback_table()
    
    try:
        with engine.connect() as conn:
            # Insert feedback message record
            insert_query = """
            INSERT INTO customer_feedback (feedback_message)
            VALUES (:message)
            RETURNING feedback_id;
            """
            result = conn.execute(
                text(insert_query),
                {"message": request.feedback_message}
            )
            conn.commit()
            
            feedback_id = result.fetchone()[0]
            
            return FeedbackResponse(
                status="success",
                message="Feedback message saved successfully",
                feedback_id=feedback_id
            )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error saving feedback message: {str(e)}"
        )


@feedback_router.get("/feedback-stats")
async def get_feedback_stats():
    """
    Endpoint to retrieve feedback statistics
    Returns average rating, total feedbacks, and rating distribution
    """
    try:
        with engine.connect() as conn:
            # Check if feedback table exists
            check_table = """
            SELECT EXISTS(
                SELECT 1 FROM information_schema.tables 
                WHERE table_name = 'customer_feedback'
            );
            """
            table_exists = conn.execute(text(check_table)).fetchone()[0]
            
            if not table_exists:
                return {
                    "status": "no_data",
                    "message": "No feedback data available yet",
                    "average_rating": None,
                    "total_feedbacks": 0,
                    "rating_distribution": {}
                }
            
            # Get statistics
            stats_query = """
            SELECT 
                AVG(feedback_value)::NUMERIC(3,2) as avg_rating,
                COUNT(*) as total_feedbacks,
                feedback_value,
                COUNT(*) as count
            FROM customer_feedback
            GROUP BY feedback_value
            ORDER BY feedback_value;
            """
            results = conn.execute(text(stats_query)).fetchall()
            
            if not results:
                return {
                    "status": "no_data",
                    "message": "No feedback data available yet",
                    "average_rating": None,
                    "total_feedbacks": 0,
                    "rating_distribution": {}
                }
            
            avg_rating = results[0][0] if results[0][0] else 0
            total = results[0][1]
            
            # Build rating distribution
            distribution = {}
            for row in results:
                distribution[str(row[2])] = row[3]
            
            return {
                "status": "success",
                "average_rating": float(avg_rating),
                "total_feedbacks": total,
                "rating_distribution": distribution
            }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving feedback stats: {str(e)}"
        )
