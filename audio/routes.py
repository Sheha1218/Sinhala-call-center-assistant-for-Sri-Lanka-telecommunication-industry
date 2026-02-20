import traceback
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from audio.start import start
from audio.second import second

audio_router = APIRouter()


class StartAudioRequest(BaseModel):
    connection_number: Optional[str] = None
    customer_nic: Optional[str] = None
    customer_name: Optional[str] = None


# NOTE: plain `def` so FastAPI runs it in a thread-pool
#       (start / second block on mic + playback)
@audio_router.post("/start-audio")
def start_audio(request: StartAudioRequest):
    """
    Complete audio workflow triggered by the UI button.

    Flow
    ────
    1. start.py  → Play Sinhala greeting (Google TTS)
                 → Record customer response (45 s, sounddevice)
                 → Transcribe via OpenAI Whisper
    2. second.py → Play second prompt asking for name / NIC / connection
                 → Record customer answer (30 s)
                 → Transcribe via Whisper
                 → Regex extraction  (name, NIC, connection number)
                 → Verify against PostgreSQL database
                 → Send latest transcription to WORKFLOW_API (LLM)
    """
    try:
        # ── Step 1 ───────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("STEP 1: Playing greeting + capturing response (45 s)")
        print("=" * 60)

        start_instance = start(
            connection_number=request.connection_number,
            customer_nic=request.customer_nic,
            customer_name=request.customer_name,
        )

        first_transcription = start_instance.captured_transcription or ""
        print(f"Step 1 transcription: {first_transcription}")

        # ── Step 2 ───────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("STEP 2: Playing prompt → record → extract → verify → LLM")
        print("=" * 60)

        second_instance = second(
            connection_number=request.connection_number,
            customer_nic=request.customer_nic,
            customer_name=request.customer_name,
        )

        # ── Build response ───────────────────────────────────────
        return {
            "status": "success",
            "message": "Complete audio workflow finished",
            "workflow_results": {
                "step_1": {
                    "greeting": "ආයුබෝවන් ABC ආයතනයේ මම වෙනෝරා කෙසෙද මා ඔබට සහයවෙන්නේ",
                    "transcription": first_transcription,
                    "recording_duration_s": 45,
                },
                "step_2": {
                    "prompt": "මට ඔබේ නම සහ හැදුරුම්පත් අංකය සහ සබදතා අංකය පැවසිය හැකිද",
                    "transcription": second_instance.captured_transcription,
                    "recording_duration_s": 30,
                },
                "extraction": {
                    "name": second_instance.extracted_name,
                    "nic": second_instance.extracted_nic,
                    "connection_number": second_instance.extracted_connection_number,
                },
                "verification": second_instance.verification_result,
                "llm_response": second_instance.llm_response,
            },
        }

    except Exception as e:
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"Error in audio workflow: {str(e)}",
        }
