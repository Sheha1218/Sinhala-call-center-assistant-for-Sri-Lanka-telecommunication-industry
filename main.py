from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path


from workflow.model import workflow
from TTS.tts import TTS_route
from stt.sst import STT_route
from db.verification import verification_router
from filters.name_nic import name_nic_router
from audio.routes import audio_router
from feedback.feedback import feedback_router


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Mount static directory so static files (index.html) are served
app.mount("/static", StaticFiles(directory="static"), name="static")
static = Jinja2Templates(directory='static')


@app.get("/", response_class=HTMLResponse)
async def root():
    # Serve the React single-file UI from static/index.html
    return Path("static/index.html").read_text(encoding="utf-8")


app.include_router(workflow)
app.include_router(TTS_route)
app.include_router(STT_route)
app.include_router(verification_router)
app.include_router(name_nic_router)
app.include_router(audio_router)
app.include_router(feedback_router)














if __name__ == '__main__':
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)