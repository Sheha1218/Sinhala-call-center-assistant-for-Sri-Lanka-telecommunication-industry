from typing import Dict, List


_ai_response_store: Dict[str, List[str]] = {}
_customer_message_store: Dict[str, List[str]] = {}


def save_customer_message(session_id: str, customer_message: str) -> None:
    """Store customer message (transcript) for a session."""
    if session_id not in _customer_message_store:
        _customer_message_store[session_id] = []
    if customer_message and str(customer_message).strip():
        _customer_message_store[session_id].append(str(customer_message).strip())


def get_customer_messages(session_id: str) -> List[str]:
    """Get all customer messages for a session."""
    if session_id not in _customer_message_store:
        return []
    return list(_customer_message_store[session_id])


def get_customer_messages_combined(session_id: str) -> str:
    """Get all customer messages as newline-separated string."""
    msgs = get_customer_messages(session_id)
    return "\n".join(msgs) if msgs else ""


def save_ai_response(session_id: str, ai_response: str) -> None:
    if session_id not in _ai_response_store:
        _ai_response_store[session_id] = []
    if ai_response and ai_response.strip():
        _ai_response_store[session_id].append(ai_response.strip())


def get_ai_responses(session_id: str) -> str:
    """
    Get all AI responses for a session as a single string (newline-separated).
    Returns empty string if no responses.
    """
    if session_id not in _ai_response_store:
        return ""
    return "\n".join(_ai_response_store[session_id])


def get_last_ai_response(session_id: str) -> str:
    """Get the most recent AI response for a session."""
    if session_id not in _ai_response_store or not _ai_response_store[session_id]:
        return ""
    return _ai_response_store[session_id][-1]


def clear_session(session_id: str) -> None:
    """Clear stored AI responses and customer messages for a session (e.g. after feedback submitted)."""
    if session_id in _ai_response_store:
        del _ai_response_store[session_id]
    if session_id in _customer_message_store:
        del _customer_message_store[session_id]
