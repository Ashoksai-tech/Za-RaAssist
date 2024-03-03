import pyttsx3


engine = pyttsx3.init("sapi5")
voices = engine.getProperty('voices')
engine.setProperty("voices",voices[0].id)
engine.setProperty('rate',200)



def say(text):
    engine = pyttsx3.init("sapi5")
    voices = engine.getProperty('voices')
    engine.setProperty("voices",voices[0].id)
    engine.setProperty('rate',200)
    print(f"ZaRa  A I:{text}")
    engine.say(text=text)
    engine.runAndWait()
    print("    ")


say("  ")