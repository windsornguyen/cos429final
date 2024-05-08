# Import necessary libraries
from gtts import gTTS
from IPython.display import Audio


# Function to convert text to speech and automatically play it
def text_to_speech_and_play(text):
    tts = gTTS(text, lang='en')  # Create a gTTS object
    tts.save('output.mp3')  # Save the audio file as 'output.mp3'
    return Audio('output.mp3', autoplay=True)  # Play the audio file


# Example usage
text_to_speech_and_play('helen keller has a dog named max')
