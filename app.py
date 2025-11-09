<<<<<<< HEAD
from flask import Flask, render_template, request
import whisper
from werkzeug.utils import secure_filename
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

app = Flask(__name__)

from sliding_window_summarizer import process_audio

# Load models once
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "audio" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        audio = request.files["audio"]
        if audio.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        filename = secure_filename(audio.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        audio.save(filepath)

        try:
            # Process audio using your pipeline
            results = process_audio(filepath)
            return render_template("result.html", results=results)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return render_template("index.html")


# ---------------------------
# Run app
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
