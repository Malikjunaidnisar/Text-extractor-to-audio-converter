import easyocr
from pdf2image import convert_from_path
import numpy as np
import cv2
from gtts import gTTS
from pydub import AudioSegment
import os
import whisper
import streamlit as st

whisper_model = whisper.load_model("base")  # Load Whisper model

def process_document(file, target_language):
    try:
        # Convert PDF to Images
        images = convert_from_path(file.name)

        # Extract Text from Images
        reader = easyocr.Reader(['en', 'ur'])  # Specify the language(s)
        extracted_text = []

        for image in images:
            # Convert PIL Image to NumPy array
            image_np = np.array(image)
            # Convert RGB to BGR (OpenCV format)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            result = reader.readtext(image_bgr)
            extracted_text.extend([res[1] for res in result])  # Extract only the text

        if not extracted_text:
            return "Error: No text extracted from the document.", None

        # Convert extracted text to audio and store the file paths
        audio_file_paths = []
        for i, text in enumerate(extracted_text):
            tts = gTTS(text=text, lang='ur')  # Specify the language as Urdu
            audio_file_path = f"output_{i}.mp3"
            tts.save(audio_file_path)
            audio_file_paths.append(audio_file_path)

        # Combine all audio segments into one
        combined_audio = AudioSegment.empty()
        for audio_file_path in audio_file_paths:
            segment = AudioSegment.from_mp3(audio_file_path)
            combined_audio += segment

        # Export the combined audio to a single file
        combined_audio_file_path = "combined_output.mp3"
        combined_audio.export(combined_audio_file_path, format="mp3")

        # Translate combined audio to desired language
        translated_audio_file_path = translate_audio(combined_audio_file_path, target_language)

        # Join extracted text into a single string for display
        extracted_text_string = "\n".join(extracted_text)

        return translated_audio_file_path, extracted_text_string

    except Exception as e:
        return f"An error occurred: {str(e)}", None

def translate_audio(audio_file_path, target_language):
    # Load the audio file for translation
    audio = whisper.load_audio(audio_file_path)
    audio = whisper.pad_or_trim(audio)

    # Make a prediction
    result = whisper_model.transcribe(audio)

    # Get the text from the result
    text = result["text"]

    # Translate the text to the desired language using Google Text-to-Speech
    tts = gTTS(text=text, lang=target_language)
    translated_audio_file_path = "translated_output.mp3"
    tts.save(translated_audio_file_path)

    return translated_audio_file_path

# Streamlit UI
st.title("PDF to Audio Translator")
st.write("Upload a PDF document to extract text and convert it to audio in the specified language.")

uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])
target_language = st.text_input("Target Language Code (e.g., 'en' for English, 'fr' for French)")

if st.button("Process Document"):
    if uploaded_file and target_language:
        translated_audio_path, extracted_text = process_document(uploaded_file, target_language)
        if translated_audio_path:
            st.audio(translated_audio_path)
            st.text_area("Extracted Text", value=extracted_text, height=300)
        else:
            st.error("Error processing document.")
    else:
        st.error("Please upload a PDF file and specify a target language.")
