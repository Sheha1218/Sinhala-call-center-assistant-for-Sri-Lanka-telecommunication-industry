from fastapi import FastAPI,Request
from fastapi.responses import HTMLResponse,StreamingResponse
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from audio.play import first,second,third
from workflow.model import modeloutput
from db.verification import verification


app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origin=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
static = Jinja2Templates(directory='static')

s_first=first()
S_second = second()
s_third=third()
model =modeloutput()
verification=verification()


def audio1():
    audio1 = s_first()
        
    def first_audio():
        s_first.play(r'audio\Shehan 1.mp3')



def verification_loop():
    if verification.valid == True:
        return 
    else:
        S_second.play(r'audio\Shehan 2.mp3')
        
        


class chatrequest(BaseModel):
    message:str
    
class response(BaseModel):
    reply:str
    

@app.post('/chat',response_model=response)
def chat(req:chatrequest):
    response = model.message(req.message)
    return {"reply":response}
    
    
    














if __name__ == '__main__':
    app.run(debug=True,)