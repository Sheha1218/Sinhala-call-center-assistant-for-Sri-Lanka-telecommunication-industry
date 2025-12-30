import pygame
import time


class first:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
    
    def play(self,path):
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
            
class second:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
    
    def play(self,path):
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
            
class third:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
    
    def play(self,path):
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        



        

        
        