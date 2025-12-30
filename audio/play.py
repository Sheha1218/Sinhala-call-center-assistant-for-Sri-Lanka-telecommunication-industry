import pygame
import time


class first:
    def __init__(self):
        self.sound0=pygame.init()
        self.sound0=pygame.mixer.init()
        self.sound0=pygame.mixer.music.load(r'audio\Shehan 1.mp3')
        self.sound0=pygame.mixer.music.play()
        
        
        while pygame.mixer.music.get_busy():
            time.sleep()
            
class second:
    def __init__(self):
        self.sound1=pygame.init()
        self.sound1=pygame.mixer.init()
        self.sound1=pygame.mixer.music.load(r'audio\Shehan 2.mp3')
        self.sound1=pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            time.sleep()
            
class third:
    def __init__(self):
        self.sound2=pygame.init()
        self.sound2=pygame.mixer.init()
        self.sound2=pygame.mixer.music.load(r'audio\Shehan 3.mp3')
        self.sound2=pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            time.sleep()
        



        

        
        