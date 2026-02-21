"""
Resume Parser Microservice with Local LLM
FINAL VERSION - With correct model path
"""

import os
import json
import tempfile
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import PyPDF2
import docx2txt
from llama_cpp import Llama
import re

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_EXTENSIONS'] = ['.pdf', '.docx', '.doc', '.txt']

# ABSOLUTE PATH - Use the exact path from your diagnostic
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'resume_parser', 'models', 'qwen2.5-1.5b-instruct-q4_k_m.gguf')

print("\n" + "="*60)
print("🚀 RESUME PARSER MICROSERVICE")
print("="*60)
print(f"📂 Base directory: {BASE_DIR}")
print(f"📁 Model path: {MODEL_PATH}")

# Verify model exists
if os.path.exists(MODEL_PATH):
    model_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # Convert to MB
    print(f"✅ Model found! Size: {model_size:.1f} MB")
else:
    print(f"❌ ERROR: Model not found at: {MODEL_PATH}")
    print("Please check the path and try again.")
    exit(1)

print("\n⏳ Loading model (this may take 30-60 seconds)...")

try:
    # Initialize the LLM with explicit CPU settings
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,  # Context window
        n_threads=6,  # Adjust based on your CPU
        n_gpu_layers=0,  # Force CPU only
        verbose=False,   # Set to True for debugging
        use_mmap=True,   # Use memory mapping for faster loading
        use_mlock=False  # Don't lock memory
    )
    print("✅ Model loaded successfully!")
    MODEL_LOADED = True
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    print("\nTroubleshooting tips:")
    print("1. Make sure you have enough RAM (at least 4GB free)")
    print("2. Try setting verbose=True to see detailed error")
    print("3. Check if the model file is corrupted")
    MODEL_LOADED = False
    llm = None

print("="*60 + "\n")

def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"PDF extraction error: {e}")
    return text

def extract_text_from_docx(file_path):
    """Extract text from DOCX file"""
    try:
        return docx2txt.process(file_path)
    except Exception as e:
        print(f"DOCX extraction error: {e}")
        return ""

def extract_text_from_txt(file_path):
    """Extract text from TXT file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"TXT extraction error: {e}")
        return ""

def extract_text(file_path):
    """Extract text based on file extension"""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext in ['.docx', '.doc']:
        return extract_text_from_docx(file_path)
    elif ext == '.txt':
        return extract_text_from_txt(file_path)
    else:
        return ""

def clean_text(text):
    """Clean and normalize extracted text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\-\@\+]', '', text)
    return text.strip()


def parse_with_llm(text):
    """Use local LLM to extract structured information"""

    if llm is None:
        return {"error": "LLM model not loaded"}

    # Truncate text to fit model's context window
    max_chars = 2000  # Increased from 1500
    if len(text) > max_chars:
        text = text[:max_chars] + "..."

    # Create a structured prompt for the LLM
    prompt = f"""Extract the following information from this resume and return COMPLETE valid JSON.
The candidate's name is typically written in ALL CAPS and appears near the top or bottom of the resume.
Ignore names mentioned in references or supervisors - look for the actual candidate's name in ALL CAPS.

Required fields:
- name: full name of the candidate (look for ALL CAPS name like "CLEOPATRA MAKUDO")
- email: email address
- phone: phone number
- skills: list of technical skills (exclude hobbies like "watching soccer")
- education: list of education entries with qualification, institution, year
- experience: list of work experiences with job title, company, duration
- objectives: career objectives if mentioned
- location: address or location
- languages: list of languages known

Resume text:
{text}

Return a COMPLETE JSON object with ALL fields. Do not truncate. The JSON must be properly closed with matching braces."""

    try:
        # Generate response from local LLM with even larger max_tokens
        response = llm(
            prompt,
            max_tokens=3072,  # Increased from 2048 to 3072
            temperature=0.1,
            stop=None,  # Remove stop to prevent early truncation
            echo=False
        )

        # Extract JSON from response
        output = response['choices'][0]['text'].strip()

        # Try to find and complete JSON
        try:
            # Find the start of JSON
            json_start = output.find('{')
            if json_start == -1:
                raise ValueError("No JSON object found")

            # Extract from first { to end
            json_str = output[json_start:]

            # Count braces to ensure completion
            brace_count = 0
            in_string = False
            escape_next = False
            last_valid_end = 0

            for i, char in enumerate(json_str):
                if escape_next:
                    escape_next = False
                    continue

                if char == '\\' and in_string:
                    escape_next = True
                    continue

                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue

                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            last_valid_end = i + 1

            # If we have a complete JSON object
            if last_valid_end > 0:
                json_str = json_str[:last_valid_end]

            # Clean up common JSON issues
            json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
            json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
            json_str = re.sub(r'}\s*{', '},{', json_str)  # Fix multiple objects

            # Parse the JSON
            parsed_data = json.loads(json_str)

            # Post-process to clean up extracted data
            if 'skills' in parsed_data and isinstance(parsed_data['skills'], list):
                # Remove hobbies from skills
                hobbies = ['watching soccer', 'choir', 'soccer', 'football', 'singing']
                parsed_data['skills'] = [s for s in parsed_data['skills']
                                         if s.lower() not in hobbies]

            return parsed_data

        except json.JSONDecodeError as e:
            # If parsing fails, return helpful error with the raw output
            return {
                "error": "JSON parsing failed",
                "error_details": str(e),
                "raw_extraction": output[:1000]  # Return more of the raw output
            }

    except Exception as e:
        return {"error": f"LLM inference failed: {str(e)}"}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if MODEL_LOADED else 'degraded',
        'model_loaded': MODEL_LOADED,
        'model_path': MODEL_PATH,
        'model_size_mb': round(os.path.getsize(MODEL_PATH) / (1024 * 1024), 1) if os.path.exists(MODEL_PATH) else 0,
        'message': 'Resume parser microservice is running'
    })

@app.route('/parse', methods=['POST'])
def parse_resume():
    """Parse a resume file"""
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Check file extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in app.config['UPLOAD_EXTENSIONS']:
        return jsonify({
            'error': f'Unsupported file type. Supported: {app.config["UPLOAD_EXTENSIONS"]}'
        }), 400

    # Save file temporarily
    filename = secure_filename(file.filename)
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, filename)

    try:
        file.save(file_path)

        # Extract text from file
        raw_text = extract_text(file_path)
        if not raw_text:
            return jsonify({'error': 'Could not extract text from file'}), 400

        # Clean the extracted text
        cleaned_text = clean_text(raw_text)

        # Parse with local LLM
        parsed_data = parse_with_llm(cleaned_text)

        # Add metadata
        response = {
            'success': True,
            'filename': filename,
            'file_type': ext,
            'data': parsed_data,
            'text_length': len(cleaned_text)
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        # Clean up temporary files
        try:
            os.remove(file_path)
            os.rmdir(temp_dir)
        except:
            pass

@app.route('/parse-text', methods=['POST'])
def parse_text():
    """Parse raw text content"""
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 503

    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text']
    cleaned_text = clean_text(text)
    parsed_data = parse_with_llm(cleaned_text)

    return jsonify({
        'success': True,
        'data': parsed_data,
        'text_length': len(cleaned_text)
    })

@app.route('/debug', methods=['GET'])
def debug_info():
    """Debug endpoint to check system info"""
    import sys
    import platform

    return jsonify({
        'python_version': sys.version,
        'platform': platform.platform(),
        'current_dir': os.getcwd(),
        'model_path': MODEL_PATH,
        'model_exists': os.path.exists(MODEL_PATH),
        'model_loaded': MODEL_LOADED,
        'files_in_dir': os.listdir('.')[:10]  # First 10 files
    })

if __name__ == '__main__':
    if not MODEL_LOADED:
        print("\n⚠️  WARNING: Model failed to load!")
        print("   Check the debug endpoint at http://localhost:5000/debug")
        print("   Common issues:")
        print("   - Insufficient RAM (need at least 4GB free)")
        print("   - Corrupted model file")
        print("   - Missing C++ runtime")
    else:
        print("\n✅ Server ready to accept requests!")

    print("\n📡 Server starting on http://localhost:5000")
    print("📝 Available endpoints:")
    print("   - GET  /health     - Health check")
    print("   - GET  /debug      - Debug information")
    print("   - POST /parse      - Upload resume file")
    print("   - POST /parse-text - Send raw text")
    print("\n" + "="*60)

    app.run(host='0.0.0.0', port=5000, debug=False)