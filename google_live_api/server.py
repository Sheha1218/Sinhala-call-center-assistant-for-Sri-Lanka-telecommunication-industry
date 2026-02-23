import asyncio
import base64
import json
import logging
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


load_dotenv(Path(__file__).parent.parent / ".env")

from google_live_api.agent import root_agent
from feedback.session_store import save_ai_response

logger = logging.getLogger(__name__)

# Session service for Live API
session_service = InMemorySessionService()
runner = Runner(app_name="sinhala-call-center", agent=root_agent, session_service=session_service)

live_router = APIRouter(prefix="/live", tags=["live"])


def _extract_agent_text(event) -> str | None:
    """Extract text content from agent event (content.parts or output_transcription)."""
    d = event.model_dump(exclude_none=True, by_alias=True) if hasattr(event, "model_dump") else (event if isinstance(event, dict) else {})
    text_parts = []
    content = d.get("content") or {}
    parts = content.get("parts") or []
    for p in parts:
        if isinstance(p, dict) and p.get("text"):
            text_parts.append(p["text"])
        elif hasattr(p, "text") and p.text:
            text_parts.append(p.text)
    if text_parts:
        return " ".join(text_parts).strip()
    out_trans = d.get("output_transcription") or d.get("outputTranscription")
    if out_trans:
        return out_trans.get("text", out_trans) if isinstance(out_trans, dict) else str(out_trans)
    return None


@live_router.websocket("/ws/{user_id}/{session_id}")
async def websocket_live(
    websocket: WebSocket,
    user_id: str,
    session_id: str,
) -> None:
    await websocket.accept()
    logger.info(f"Live WebSocket connected: user_id={user_id}, session_id={session_id}")

    model_name = root_agent.model or ""
    is_native_audio = "native-audio" in model_name.lower()

    if is_native_audio:
        run_config = RunConfig(
            streaming_mode=StreamingMode.BIDI,
            response_modalities=["AUDIO"],
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            session_resumption=types.SessionResumptionConfig(),
        )
    else:
        run_config = RunConfig(
            streaming_mode=StreamingMode.BIDI,
            response_modalities=["TEXT"],
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            session_resumption=types.SessionResumptionConfig(),
        )

    session = await session_service.get_session(
        app_name="sinhala-call-center", user_id=user_id, session_id=session_id
    )
    if not session:
        await session_service.create_session(
            app_name="sinhala-call-center", user_id=user_id, session_id=session_id
        )

    live_request_queue = LiveRequestQueue()

    async def upstream_task():
        while True:
            msg = await websocket.receive()
            if "bytes" in msg:
                audio_data = msg["bytes"]
                blob = types.Blob(mime_type="audio/pcm;rate=16000", data=audio_data)
                live_request_queue.send_realtime(blob)
            elif "text" in msg:
                try:
                    data = json.loads(msg["text"])
                    if data.get("type") == "text":
                        content = types.Content(parts=[types.Part(text=data["text"])])
                        live_request_queue.send_content(content)
                except json.JSONDecodeError:
                    pass

    async def downstream_task():
        async for event in runner.run_live(
            user_id=user_id,
            session_id=session_id,
            live_request_queue=live_request_queue,
            run_config=run_config,
        ):
            event_json = event.model_dump_json(exclude_none=True, by_alias=True) if hasattr(event, "model_dump_json") else json.dumps({})

            # Save AI response to session store for feedback linking
            author = getattr(event, "author", None)
            if author and author != "user":
                text = _extract_agent_text(event)
                if text:
                    save_ai_response(session_id, text)

            await websocket.send_text(event_json)

    try:
        await asyncio.gather(upstream_task(), downstream_task())
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Live streaming error: {e}", exc_info=True)
    finally:
        live_request_queue.close()


@live_router.get("/session-id")
async def get_session_id():
    """Generate a new session ID for voice conversation (used by feedback linking)."""
    return {"session_id": str(uuid.uuid4())}
