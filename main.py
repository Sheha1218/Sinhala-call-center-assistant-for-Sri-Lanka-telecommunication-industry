from fastapi import FastAPI,Request
from fastapi.responses import HTMLResponse,StreamingResponse
from fastapi.templating import Jinja2Templates
from audio.play import first,second,third
import pygame

app=FastAPI()
static = Jinja2Templates(directory='static')

s_first=first
S_second = second
s_third=third

class main:
    def __init__(self):
        self.audio = s_first()
        
    def first_audio(self):
        self.audio.play(r'audio\Shehan 1.mp3')

main().first_audio()
        




if __name__ == '__main__':
    app.run(debug=True,)