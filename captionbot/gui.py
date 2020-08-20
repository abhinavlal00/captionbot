# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 11:28:42 2020

@author: ABHINAV
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 08:34:13 2020

@author: ABHINAV
"""

import tensorflow
import gtts
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
import time
from captiongen import generateCaption,encodeImage
from kivy.core.audio import SoundLoader
Builder.load_string('''
<CameraClick>:
    orientation: 'vertical'
    Camera:
        id: camera
        resolution: (640, 480)
        play: False
    ToggleButton:
        text: 'Play'
        on_press: camera.play = not camera.play
        size_hint_y: None
        height: '48dp'
    Button:
        text: 'Capture'
        size_hint_y: None
        height: '48dp'
        on_press: root.capture()
    Button:
        id: btn
        text: self.text
        size_hint_y: None
        height: '48dp'
        on_press: 'Capture for next caption.'
''')


class CameraClick(BoxLayout):
    def capture(self):
        camera = self.ids['camera']
        timestr = time.strftime("%Y%m%d_%H%M%S")
        camera.export_to_png("IMG_{}.jpg".format(timestr))
        
        print("Captured")
        img = tensorflow.keras.preprocessing.image.load_img("IMG_{}.jpg".format(timestr), target_size=(299, 299))
        img=encodeImage(img)
        img=img.reshape((1,2048))
        capti=generateCaption(img)
        print(capti)
        #Label(text=capti, font_size='20sp')
        self.ids['btn'].text=capti


class TestCamera(App):

    def build(self):
        return CameraClick()


TestCamera().run()