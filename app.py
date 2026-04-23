# app.py
# This is the main web server using Flask
# It connects everything together

import os
import uuid
from flask import Flask, request, jsonify, render_template, session
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from rag_engine import load_csv_to_vectorstore, get_dataset_summary
from agents import run_multi_agent_pipeline, clear_memory
from guardrails import check_prompt_injection, sanitize_input

load_dotenv()

# Create the Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Configure file upload settings
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"csv"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if uploaded file is a CSV"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    """Serve the main web page"""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    """Handle CSV file upload and load into vector store"""
    
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files["file"]
    
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "Only CSV files are allowed"}), 400
    
    # Save the file securely
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4()}_{filename}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
    file.save(filepath)
    
    # Load CSV into vector store
    try:
        summary = load_csv_to_vectorstore(filepath)
        clear_memory()  # Reset conversation for new dataset
        
        return jsonify({
            "success": True,
            "message": f"✅ Dataset loaded successfully!",
            "summary": {
                "rows": summary["rows"],
                "columns": summary["columns"],
                "null_count": summary["null_count"]
            }
        })
    except Exception as e:
        return jsonify({"error": f"Failed to process file: {str(e)}"}), 500


@app.route("/chat", methods=["POST"])
def chat():
    """Handle user chat messages"""
    
    data = request.json
    user_message = data.get("message", "").strip()
    
    if not user_message:
        return jsonify({"error": "Empty message"}), 400
    
    # Check if data is loaded
    summary = get_dataset_summary()
    if not summary:
        return jsonify({
            "error": "Please upload a CSV file first before asking questions."
        }), 400
    
    # GUARDRAIL: Sanitize input
    user_message = sanitize_input(user_message)
    
    # GUARDRAIL: Check for prompt injection and off-topic questions
    is_safe, reason = check_prompt_injection(user_message)
    if not is_safe:
        return jsonify({
            "analyst_output": "🔒 Blocked by safety guardrail",
            "explainer_output": reason,
            "blocked": True
        })
    
    # Run the multi-agent pipeline
    try:
        result = run_multi_agent_pipeline(user_message)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Agent error: {str(e)}"}), 500


if __name__ == "__main__":
    print("🚀 Starting Data Explainer AI...")
    print("📊 Open your browser and go to: http://localhost:5000")
    app.run(debug=True, port=5000)