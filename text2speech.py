import pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 115)     # setting up new voice rate
engine.say("I will speak this text")
engine.runAndWait()

from gtts import gTTS
import os

speech_text="Hello, I am a text to speech program. I convert text to speech. I can speak multiple languages. I can also speak in different accents. I can also speak in different speeds. I can also speak in different pitches. I can also speak in different volumes. I can also speak in different emotions. I can also speak in"
speech = gTTS(text=speech_text, lang='en')
speech.save("output.mp3")
os.system("mpg321 output.mp3")  # Plays the audio