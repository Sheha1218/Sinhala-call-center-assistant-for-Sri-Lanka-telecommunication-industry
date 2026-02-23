from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import time

from workflow.model import workflow
from feedback.feedback import feedback_router
from google_live_api.server import live_router

app = FastAPI()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    print(f"[API] {request.method} {request.url.path} -> {response.status_code} ({elapsed:.2f}s)")
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Mount static directory for static files (feedback.html, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")
static = Jinja2Templates(directory='static')


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
@app.get("/voice", response_class=HTMLResponse)
async def root():
    
    return Path("static/live-voice.html").read_text(encoding="utf-8")


app.include_router(workflow)
app.include_router(feedback_router)
app.include_router(live_router)



