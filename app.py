# --------- FIXES for Windows & ffmpeg compatibility ---------
import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import os
os.environ["PATH"] += os.pathsep + r"C:\Users\HP\Downloads\ffmpeg-7.1.1-full_build\ffmpeg-7.1.1-full_build\bin"

# --------- Import libraries ---------
import whisper
from transformers import pipeline
import spacy
import re
import streamlit as st
from deepface import DeepFace
import cv2

# --------- Step 1: Transcription ---------
@st.cache_data
def transcribe_audio(audio_path):
    if not os.path.exists(audio_path):
        st.error(f"File not found: {audio_path}")
        return ""

    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]

# --------- Step 2: Summarization with chunking ---------
@st.cache_data
def summarize_text(text, max_chunk_length=1000):
    if not text or len(text.strip()) < 20:
        return "❌ Transcript is too short to summarize."

    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Split text into chunks of max_chunk_length characters
    text_chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]

    summaries = []
    for chunk in text_chunks:
        try:
            summary = summarizer(chunk, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
            summaries.append(summary)
        except Exception as e:
            summaries.append("[Summary failed for a chunk]")
            st.warning(f"Chunk summarization error: {e}")

    final_summary = " ".join(summaries)
    return final_summary

# --------- Step 3: Extract Action Items (broader regex) ---------
def extract_action_items(text):
    pattern = r"(?i)(action item|task|to do|follow up)[\:\-\s]*(.+?)(\.|\n|$)"
    matches = re.findall(pattern, text)
    return [m[1].strip() for m in matches if m[1].strip()]

# --------- Step 4: Extract Participants ---------
@st.cache_data
def extract_names(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    names = list(set([ent.text for ent in doc.ents if ent.label_ == "PERSON"]))
    return names

# --------- Step 5: Assign Tasks to Participants ---------
def assign_tasks_to_people(action_items, participants):
    assignments = {}
    for item in action_items:
        assigned = [name for name in participants if name in item]
        assignments[item] = assigned[0] if assigned else "Unassigned"
    return assignments

# --------- Optional Step: Analyze emotions from video ---------
def analyze_emotions_from_video(video_path, sample_rate=30):
    emotions = []
    vidcap = cv2.VideoCapture(video_path)
    success, frame = vidcap.read()
    count = 0
    while success:
        if count % sample_rate == 0:
            frame_path = f"frame_{count}.jpg"
            cv2.imwrite(frame_path, frame)
            try:
                analysis = DeepFace.analyze(frame_path, actions=['emotion'], enforce_detection=False)
                dominant_emotion = analysis.get('dominant_emotion', "Unknown")
            except Exception:
                dominant_emotion = "Unknown"
            emotions.append(dominant_emotion)
            os.remove(frame_path)
        success, frame = vidcap.read()
        count += 1
    vidcap.release()
    return emotions

# --------- Streamlit UI ---------
st.title("📝 AI Meeting Summarizer & Action Item Tracker")

uploaded_audio = st.file_uploader("Upload Meeting Audio (.wav or .mp3)", type=["wav", "mp3"])
uploaded_video = st.file_uploader("Upload Meeting Video (.mp4) for Emotion Analysis (optional)", type=["mp4"])

if uploaded_audio:
    audio_path = "temp_audio.wav"
    with open(audio_path, "wb") as f:
        f.write(uploaded_audio.getbuffer())

    st.info("Transcribing audio...")
    transcript = transcribe_audio(audio_path)
    st.subheader("Transcript")
    st.write(transcript)

    st.info("Summarizing meeting...")
    summary = summarize_text(transcript)
    st.subheader("Summary")
    st.write(summary)

    st.info("Extracting action items...")
    action_items = extract_action_items(transcript)
    if action_items:
        st.subheader("Action Items Found")
        for idx, item in enumerate(action_items, 1):
            st.write(f"{idx}. {item}")
    else:
        st.write("No clear action items detected.")

    st.info("Detecting participants...")
    participants = extract_names(transcript)
    if participants:
        st.subheader("Participants Detected")
        st.write(", ".join(participants))
    else:
        st.write("No participant names detected.")

    st.info("Assigning tasks...")
    assignments = assign_tasks_to_people(action_items, participants)
    if assignments:
        st.subheader("Task Assignments")
        for task, person in assignments.items():
            st.write(f"- {task} → **Assigned to:** {person}")

    if uploaded_video:
        video_path = "temp_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        st.info("Analyzing emotions from video...")
        emotions = analyze_emotions_from_video(video_path)
        if emotions:
            st.subheader("Detected Emotions from Video")
            st.write(emotions)
        else:
            st.write("No emotions detected.")

else:
    st.write("Please upload meeting audio to start.")
