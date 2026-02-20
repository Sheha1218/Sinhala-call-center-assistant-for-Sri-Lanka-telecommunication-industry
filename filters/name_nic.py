from fastapi import APIRouter
from pydantic import BaseModel
import re
import json

name_nic_router = APIRouter()

class ExtractNameNICRequest(BaseModel):
    transcription:str
    connection_number:str=None


class ExtractNameNICResponse(BaseModel):
    status:str
    extracted_name:str=None
    extracted_nic:str=None
    extracted_connection_number:str=None
    raw_transcription:str
    message: str=None


def extract_name_from_text(text: str) -> str:
    """
    Extract name from transcription using regex patterns
    Looks for proper names (capitalized words or Sinhala script)
    """
    if not text:
        return None
    
    # Pattern 1: Extract capitalized words (English names)
    english_name_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
    english_matches = re.findall(english_name_pattern, text)
    if english_matches:
        return english_matches[0].strip()
    
    # Pattern 2: Extract words at the beginning of sentences (potential names)
    first_word_pattern=r'^([a-zA-Z\u0D80-\u0DFF]+)'
    first_word_match=re.search(first_word_pattern, text.strip())
    if first_word_match:
        return first_word_match.group(1).strip()
    
    # Pattern 3: Any continuous alphabetic sequence (fallback)
    word_pattern=r'[a-zA-Z\u0D80-\u0DFF]+'
    words=re.findall(word_pattern, text)
    if words:
        # Return the longest word as it's likely a name
        return max(words, key=len)
    
    return None


def extract_nic_from_text(text: str) -> str:
    """
    Extract NIC (National Identity Card) number from transcription
    Sri Lankan NIC format: 9 digits + 1 letter (V, X, or space)
    Examples: 123456789V, 987654321X
    """
    if not text:
        return None
    
    # Pattern 1: NIC with letter at end (e.g., 123456789V, 987654321X)
    nic_pattern = r'\b(\d{9,12}[VXvx])\b'
    nic_match = re.search(nic_pattern, text)
    if nic_match:
        return nic_match.group(1).strip().upper()
    
    # Pattern 2: Continuous digit sequence (9+ digits)
    digit_pattern = r'\b(\d{9,12})\b'
    digit_matches = re.findall(digit_pattern, text)
    if digit_matches:
        # Return the longest sequence of digits
        return max(digit_matches, key=len)
    
    return None


def extract_connection_number_from_text(text: str) -> str:
    """
    Extract connection/telephone number from transcription
    Looks for 10-digit sequences or formatted numbers
    Examples: 0123456789, 071-234-5678, 071 234 5678
    """
    if not text:
        return None
    
    # Pattern 1: 10 consecutive digits (standard phone number)
    phone_pattern = r'\b([0-9]{10})\b'
    phone_match = re.search(phone_pattern, text)
    if phone_match:
        return phone_match.group(1).strip()
    
    # Pattern 2: Formatted phone with dashes or spaces (071-234-5678 or 071 234 5678)
    formatted_pattern = r'\b([0-9]{2,3}[-\s]?[0-9]{2,3}[-\s]?[0-9]{3,4})\b'
    formatted_matches = re.findall(formatted_pattern, text)
    if formatted_matches:
        # Return the first match, removing spaces and dashes
        return formatted_matches[0].replace('-', '').replace(' ', '')
    
    # Pattern 3: Any 7-10 digit sequence
    digit_sequence = r'\b([0-9]{7,10})\b'
    digit_matches = re.findall(digit_sequence, text)
    if digit_matches:
        # Return the first match
        return digit_matches[0]
    
    return None


@name_nic_router.post("/extract-name-nic", response_model=ExtractNameNICResponse)
async def extract_name_nic(request: ExtractNameNICRequest):
    """
    Extract name, NIC number, and connection number from transcription text
    Uses regex patterns to identify customer data
    """
    try:
        transcription = request.transcription.strip()
        
        if not transcription:
            return ExtractNameNICResponse(
                status="error",
                message="Empty transcription provided",
                raw_transcription=transcription
            )
        
        # Extract name, NIC, and connection number
        extracted_name = extract_name_from_text(transcription)
        extracted_nic = extract_nic_from_text(transcription)
        extracted_connection_number = extract_connection_number_from_text(transcription)
        
        # If connection number not extracted from transcription, use provided one
        if not extracted_connection_number and request.connection_number:
            extracted_connection_number = request.connection_number
        
        # Validate extraction
        if not extracted_name or not extracted_nic:
            return ExtractNameNICResponse(
                status="partial",
                extracted_name=extracted_name,
                extracted_nic=extracted_nic,
                extracted_connection_number=extracted_connection_number,
                raw_transcription=transcription,
                message="Could not extract complete name and NIC from transcription"
            )
        
        return ExtractNameNICResponse(
            status="success",
            extracted_name=extracted_name,
            extracted_nic=extracted_nic,
            extracted_connection_number=extracted_connection_number,
            raw_transcription=transcription,
            message=f"Successfully extracted - Name: {extracted_name}, NIC: {extracted_nic}, Connection: {extracted_connection_number}"
        )
        
    except Exception as e:
        return ExtractNameNICResponse(
            status="error",
            message=str(e),
            raw_transcription=request.transcription
        )

