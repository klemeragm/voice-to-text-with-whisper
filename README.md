# voice-to-text-with-whisper

**This project is a code to transcribe or translate audios and medias with a Whisper library.**

# 1 First: You neeed to install: 

<code>pip install whisper</code>

# install openai's whisper
<code> !pip install git+https://github.com/openai/whisper.git </code>

# update the packages
<code> !sudo apt update && sudo apt install ffmpeg </code>

# In the code:

You need to following the other imports and above the code below. 

#In the final 

You can set the:  
#- Language Detection
# detect the spoken language

<code> _,probs = model.detect_language(mel)
lang = max(probs, key=probs.get)
prob = "{0:.0%}".format(max(probs.values()))

#Print language that scored the higuest likihoood
print(f'Detected language (and Probability): {lang}', f'({prob})') </code>


# English or another to Language Translation (Example with language='pt') 

# decode the audio
options= whisper.DecodingOptions(language='pt',task='translate', fp16=False, without_timestamps=False)
result = whisper.decode(model,mel,options)

# print the Result:
print(f'The text translation is: {result.text}')

**Refs:** https://github.com/openai/whisper
https://github.com/AndrewMayneProjects/Whisper 






