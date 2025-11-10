import os
os.environ["PATH"] += os.pathsep + r"C:\Users\daksh\Downloads\ffmpeg-8.0-essentials_build\ffmpeg-8.0-essentials_build\bin"
import whisper

# load model (base or small for speed)
whisper_model = whisper.load_model("base")

from transformers import PegasusTokenizer, PegasusForConditionalGeneration, AutoTokenizer,AutoModelForSeq2SeqLM
import torch

from transformers import pipeline

device = "cpu"
pipe = pipeline("summarization", model="facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn").to(device)

AUDIO_PATH = "lecture.mp3"   # change this to your file path
CHUNK_WORDS = 210
device = "cuda" if torch.cuda.is_available() else "cpu"

def transcribe_audio(AUDIO_PATH):
  print("Transcribing audio...")
  result = whisper_model.transcribe(AUDIO_PATH)
  segments = result["segments"]
  return segments

def chunk_segments(segments, chunk_size=210):
    chunks = []
    current_chunk = {"start": segments[0]["start"], "text": ""}
    word_count = 0

    for seg in segments:
        text = seg["text"].strip()
        words = text.split()
        word_count += len(words)
        current_chunk["text"] += " " + text

        # Once chunk reaches ~chunk_size words, start a new one
        if word_count >= chunk_size:
            current_chunk["end"] = seg["end"]
            chunks.append(current_chunk)
            current_chunk = {"start": seg["start"], "text": ""}
            word_count = 0

    # add last chunk if not empty
    if current_chunk["text"].strip():
        current_chunk["end"] = segments[-1]["end"]
        chunks.append(current_chunk)

    return chunks

def summarize_text(text, min_len=50, max_len=70):
    inputs = tokenizer(
        text, truncation=True, padding="longest", return_tensors="pt", max_length=1024
    ).to(device)
    summary_ids = model.generate(
        **inputs,
        num_beams=6,
        no_repeat_ngram_size=3,
        min_length=min_len,
        max_length=max_len
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def process_audio(AUDIO_PATH):
  segments = transcribe_audio(AUDIO_PATH)
  chunks = chunk_segments(segments)

  results = []
  for chunk in chunks:
    summary = summarize_text(chunk["text"])
    results.append({
        "start": chunk["start"],
        "end": chunk["end"],
        "transcript": chunk["text"].strip(),
        "summary": summary
    })

  return results
