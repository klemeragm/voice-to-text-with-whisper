#intalar o openai-whisper
#instalar o ffmpeg: https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/

import whisper
import re
import pathlib
import os
import openai 
import json 
import torch

path = pathlib.Path(r"C:/Users/kleme/Documents/Python Scripts/WhisperAI/audio_teste.mp4")  #def main():


#from whisper.tokenizer import LANGUAGES
model = whisper.load_model("medium")  # ~10 GB 

#nome do áudio com o formato mp3 ou wav
# load audio and pad/trim it to fit 30 seconds

audio = whisper.load_audio(path) #path to file
audio = whisper.pad_or_trim(audio)

#make log-mel spectogram and move to the same device as the model
mel = whisper.log_mel_spectogram(audio).to(model.device)

result = model.transcribe(path, language='pt', fp16=False, world_timestamps=True)
result = json.loads(whisper.decode(model,mel,options))
result = result["text"][0].content()
    
# print the Result:
print(f'The text translation is: {result.text}')


"""
#if __name__ == "main()":

#entrada = input(str("Entre com nome do caminho do arquivo: {path}.{\WhisperAI\"*.m__}",path)

#print(result["text"][0].content())


# --- Other format #3.0 - Translation with Perfoming Python
  
# Load de the model
from whisper.tokenizer import LANGUAGES
model_medium = whisper.load_model("medium")

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("content/Whisper/audio_teste.mp3") #path to file
audio = whisper.pad_or_trim(audio)

#make log-mel spectogram and move to the same device as the model
mel = whisper.log_mel_spectogram(audio).to(model.device)

#3.1 - Language Detection 
_,probs = model.detect_language(mel) # detect the spoken language
lang = max(probs, key=probs.get)
prob = "{0:.0%}".format(max(probs.values()))

#Print language that scored the higuest likihoood
print(f'Detected language (and Probability): {lang}', f'({prob})')


#3.2 - English or another to Pt Translation
# decode the audio
options= whisper.DecodingOptions(language='pt', task='translate', fp16=False, without_timestamps=False)
result = whisper.decode(model,mel,options)

# print the Result:
print(f'The text translation is: {result.text}')

"""