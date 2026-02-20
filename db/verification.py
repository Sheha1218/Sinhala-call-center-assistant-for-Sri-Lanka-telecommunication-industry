import psycopg2
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

WORKFLOW_API = os.getenv('WORKFLOW_API')
db_url = os.getenv('db_url')

verification_router = APIRouter()


class VerificationRequest(BaseModel):
    connection_number: str
    customer_nic: str
    customer_name: str
    transcription: str


class ExtractedVerificationRequest(BaseModel):
    connection_number: str
    extracted_customer_name: str
    extracted_customer_nic: str
    transcription: str


class verification:
    def __init__(self):
        self.conn = psycopg2.connect(
            host="localhost",
            database='call_agent',
            user="postgres",
            password='root',
            port="5432"
        )
        self.cursor = self.conn.cursor()

    def verify_customer(self, connection_number, customer_nic, customer_name):
        """
        Verify if customer exists in the database with matching details
        Returns: (is_valid, customer_data)
        """
        query = """
                SELECT * FROM customers
                WHERE connection_number = %s AND customer_nic = %s AND customer_name = %s;
        """
        try:
            self.cursor.execute(query, (connection_number, customer_nic, customer_name))
            result = self.cursor.fetchone()
            
            if result:
                return True, result
            else:
                return False, None
        except Exception as e:
            print(f"Database error: {str(e)}")
            return False, None

    def close(self):
        self.cursor.close()
        self.conn.close()


# API endpoint to verify customer and post to WORKFLOW_API
@verification_router.post("/verify-and-process")
async def verify_and_process(request: VerificationRequest):
    """
    Verify customer details from transcription data and post to WORKFLOW_API
    """
    verifier = verification()
    
    try:
        # Verify customer against database
        is_valid, customer_data = verifier.verify_customer(
            request.connection_number,
            request.customer_nic,
            request.customer_name
        )
        
        if not is_valid:
            raise HTTPException(
                status_code=401,
                detail="Customer verification failed. Connection number, NIC, or name does not match."
            )
        
        # If verification successful, post to WORKFLOW_API with transcription
        payload = {
            "message": request.transcription,
            "customer": {
                "connection_number": request.connection_number,
                "customer_nic": request.customer_nic,
                "customer_name": request.customer_name
            }
        }
        
        response = requests.post(
            WORKFLOW_API,
            json=payload,
            timeout=60.0
        )
        response.raise_for_status()
        
        return {
            "status": "success",
            "verification": "verified",
            "message": "Customer verified and data sent to LLM API",
            "llm_response": response.text if response.text else "Processing..."
        }
        
    except requests.exceptions.RequestException as e:
        return {
            "status": "error",
            "message": f"Error posting to WORKFLOW_API: {str(e)}"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
    finally:
        verifier.close()


@verification_router.post("/verify-extracted-data")
async def verify_extracted_data(request: ExtractedVerificationRequest):
    """
    Verify customer using extracted name and NIC number, then send latest transcription to LLM
    """
    verifier = verification()
    engine = create_engine(db_url)
    
    try:
        # Verify customer using extracted data (flexible matching)
        is_valid, customer_data = verifier.verify_customer(
            request.connection_number,
            request.extracted_customer_nic,
            request.extracted_customer_name
        )
        
        if not is_valid:
            return {
                "status": "error",
                "message": f"Customer not found with name: {request.extracted_customer_name}, NIC: {request.extracted_customer_nic}"
            }
        
        # Get latest transcription from database for this customer
        latest_transcription = None
        try:
            with engine.connect() as connection:
                query = """
                    SELECT transcription FROM stt_transcriptions 
                    ORDER BY created_at DESC 
                    LIMIT 1
                """
                result = connection.execute(text(query))
                row = result.fetchone()
                if row:
                    latest_transcription = row[0]
        except Exception as db_error:
            print(f"Database error fetching transcription: {str(db_error)}")
            latest_transcription = request.transcription
        
        if not latest_transcription:
            latest_transcription = request.transcription
        
        # Send latest transcription to WORKFLOW_API
        if WORKFLOW_API:
            try:
                payload = {
                    "message": latest_transcription,
                    "customer": {
                        "connection_number": request.connection_number,
                        "customer_name": request.extracted_customer_name,
                        "customer_nic": request.extracted_customer_nic
                    }
                }
                
                llm_response = requests.post(
                    WORKFLOW_API,
                    json=payload,
                    timeout=60.0
                )
                llm_response.raise_for_status()
                
                return {
                    "status": "success",
                    "verification": "verified",
                    "customer_name": request.extracted_customer_name,
                    "customer_nic": request.extracted_customer_nic,
                    "message": "Customer verified and sent to LLM",
                    "transcription_used": latest_transcription,
                    "llm_response": llm_response.json() if llm_response.text else "Processing..."
                }
            except requests.exceptions.RequestException as e:
                return {
                    "status": "error",
                    "message": f"Error posting to WORKFLOW_API: {str(e)}"
                }
        else:
            return {
                "status": "success",
                "verification": "verified",
                "customer_name": request.extracted_customer_name,
                "customer_nic": request.extracted_customer_nic,
                "message": "Customer verified",
                "transcription_used": latest_transcription,
                "warning": "WORKFLOW_API not configured"
            }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
    finally:
        verifier.close()


