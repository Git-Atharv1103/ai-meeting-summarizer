# AI Meeting Summarizer & Action Item Tracker

This project allows you to upload meeting audio (and optionally video) files, transcribes the meeting, summarizes it, extracts action items, assigns tasks to participants, and analyzes participant emotions from video frames.

## Features

- Audio transcription using OpenAI Whisper
- Meeting summarization using Hugging Face transformers (BART)
- Action item extraction via regex
- Participant name detection using spaCy NER
- Emotion analysis from video frames using DeepFace
- Simple web UI with Streamlit

## Setup Instructions

1. Clone the repo:

```bash
git clone <repo_url>
cd ai-meeting-summarizer
