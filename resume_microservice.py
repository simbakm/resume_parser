"""
Resume Parser Microservice with Local LLM
FINAL VERSION - With correct model path
"""

import os
import json
import tempfile
import logging
import time
from datetime import datetime
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import PyPDF2
import pdfplumber
import docx2txt
from llama_cpp import Llama
import re
from download_model import download_model
from flasgger import Swagger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Log to console
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize Swagger for API documentation
swagger = Swagger(app, template={
    "swagger": "2.0",
    "info": {
        "title": "Resume Parser Microservice",
        "description": "AI-powered resume parsing with local LLM (Qwen 2.5)",
        "version": "1.0.0",
        "contact": {"name": "Resume Parser"}
    },
    "host": "simbakm-resume-parser.hf.space",
    "schemes": ["https"],
    "basePath": "/"
})


def ensure_model():
    """
    Ensure the model is downloaded and ready before accepting requests.
    This blocks startup until the model is fully available.
    """
    if not os.path.exists(MODEL_PATH):
        print("\n" + "⚠️ "*30)
        print("📥 DOWNLOADING MODEL - This may take 2-5 minutes on first startup...")
        print("⚠️ "*30)
        download_model("qwen_fast")
    else:
        print("✅ Model already exists, skipping download")

    # Verify model was downloaded successfully
    if not os.path.exists(MODEL_PATH):
        print("❌ CRITICAL ERROR: Model download failed!")
        raise RuntimeError("Model download failed. Cannot continue.")

    print("✅ Model download ready")
    return True




# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_EXTENSIONS'] = ['.pdf', '.docx', '.doc', '.txt']

# ABSOLUTE PATH - Use the exact path from your diagnostic
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'resume_parser', 'models', 'qwen2.5-0.5b-instruct-q4_k_m.gguf')

print("\n" + "="*60)
print("🚀 RESUME PARSER MICROSERVICE - STARTUP SEQUENCE")
print("="*60)
print(f"📂 Base directory: {BASE_DIR}")
print(f"📁 Model path: {MODEL_PATH}")

print("\n⏳ STEP 1: Ensuring model is available...")
# Ensure model is downloaded before anything else
ensure_model()

# Verify model exists
if os.path.exists(MODEL_PATH):
    model_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # Convert to MB
    print(f"✅ STEP 1 COMPLETE: Model verified! Size: {model_size:.1f} MB")
else:
    print(f"❌ CRITICAL ERROR: Model not found at: {MODEL_PATH}")
    exit(1)

print("\n⏳ STEP 2: Loading model into memory (this may take 30-60 seconds)...")

try:
    # Initialize the LLM with explicit CPU settings
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=8192,  # Increased context window to handle large text limits and generation
        n_threads=2,  # Optimized for 2-vCPU Hugging Face environments
        n_gpu_layers=0,  # Force CPU only
        verbose=False,   # Set to True for debugging
        use_mmap=True,   # Use memory mapping for faster loading
        use_mlock=False  # Don't lock memory
    )
    print("✅ STEP 2 COMPLETE: Model loaded successfully!")
    MODEL_LOADED = True
except Exception as e:
    print(f"❌ FAILED: Could not load model into memory: {e}")
    print("\nTroubleshooting:")
    print("1. Ensure you have enough RAM (minimum 4GB free)")
    print("2. Check disk space for temporary files")
    print("3. Try setting verbose=True to see detailed errors")
    print("4. Model file may be corrupted - try re-downloading")
    MODEL_LOADED = False
    llm = None

print("="*60 + "\n")

def extract_text_from_pdf(file_path):
    """Extract text from PDF file using pdfplumber as primary, fallback to PyPDF2"""
    text = ""
    try:
        # Primary: pdfplumber (more robust for layouts/encodings)
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        if text.strip():
            return text
            
    except Exception as e:
        logger.warning(f"pdfplumber extraction error: {e}")

    # Fallback: PyPDF2
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        logger.error(f"PyPDF2 extraction error: {e}")
        
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

def _try_fix_truncated_json(output):
    """Try to repair truncated JSON by appending closing braces."""
    start = output.find('{')
    if start == -1:
        return None
    candidate = output[start:]

    # Quick heuristic: if it already ends with }, good. else append braces
    if candidate.strip().endswith('}'):
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # Try adding up to 5 braces
    for i in range(1, 6):
        try_candidate = candidate + '}' * i
        try:
            return json.loads(try_candidate)
        except Exception:
            continue
    return None

def parse_with_llm(text):
    """Use local LLM to extract structured information with high token budget"""
    
    logger.info("   📋 Preparing LLM input...")

    if llm is None:
        logger.error("LLM model is not initialized")
        return {"error": "LLM model not loaded"}

    # Limit input to avoid token overflow, but be generous
    max_input_chars = 3000
    original_len = len(text)
    if len(text) > max_input_chars:
        logger.warning(f"   ⚠️  Text truncated from {original_len} to {max_input_chars} characters")
        text = text[:max_input_chars] + "..."
    
    logger.info(f"   ✓ Input text length: {len(text)} characters")

    prompt = f"""Task: Extract resume fields into the following JSON format.
Strict Rules: Use ONLY information from the resume. Do NOT use placeholder text.

Resume:
{text}

JSON Format:
{{
  "name": "",
  "email": "",
  "phone": "",
  "skills": [],
  "education": [
    {{
      "qualification": "",
      "institution": "",
      "year": ""
    }}
  ],
  "experience": [
    {{
      "job_title": "",
      "company": "",
      "duration": ""
    }}
  ],
  "objectives": "if not given, give a brief summary starting with To",
  "location": "full address",
  "languages": ["eg shona, english"]
}}"""

    try:
        logger.info("   🔄 Sending request to Qwen 2.5 LLM (high token budget)...")
        llm_request_start = time.time()
        
        # Use chat completion to automatically apply ChatML stops and prevent runaway output
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a highly concise JSON extraction tool. Use empty strings for missing data. Do NOT repeat or summarize long sections of text. Be extremely brief."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=768, # Tightened to prevent wordy/repeated responses
            temperature=0.1,
            stop=["<|im_end|>", "<|endoftext|>", "```"]
        )
        
        llm_request_time = time.time() - llm_request_start
        logger.info(f"   ✓ LLM response received in {llm_request_time:.2f}s")

        output = response['choices'][0]['message']['content'].strip()
        logger.info(f"   ✓ Response length: {len(output)} characters")

        try:
            logger.info("   🔍 Parsing JSON response...")
            
            # 1. Clean markdown code blocks if present
            clean_output = re.sub(r'```json\s*', '', output)
            clean_output = re.sub(r'```\s*', '', clean_output)
            clean_output = clean_output.strip()

            # 2. Find the start of the JSON object
            json_start = clean_output.find('{')
            if json_start == -1:
                logger.error("   ❌ No JSON object found in response")
                raise ValueError("No JSON object found")

            json_str = clean_output[json_start:]
            
            # 3. Find matching closing brace to handle extra text or unclosed arrays
            brace_count = 0
            in_string = False
            escape_next = False
            json_end = 0

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
                            json_end = i + 1
                            break

            if json_end > 0:
                json_str = json_str[:json_end]
            elif brace_count > 0:
                # If we never reached 0, it was truncated. Try to fix.
                logger.warning(f"   ⚠️  JSON truncated (brace count: {brace_count})")
                
                # Simple balanced fix: append closing characters
                # Check where it stopped. If in the middle of a string, close quote.
                if in_string:
                    json_str += '"'
                
                # Check for list/object structure
                # This is heuristic, but helps recovery
                if json_str.count('[') > json_str.count(']'):
                    json_str += ']'
                
                json_str += '}' * brace_count
                logger.info(f"   🔧 Applied basic balancing fix")
            
            # Clean up common JSON issues
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)

            logger.info("   📦 Validating JSON structure...")
            parsed_data = json.loads(json_str)
            logger.info("   ✓ JSON validation successful")

            # Force name to ALL CAPS as required
            if 'name' in parsed_data and isinstance(parsed_data['name'], str):
                parsed_data['name'] = parsed_data['name'].upper()
                logger.info(f"   ✓ Name formatted to ALL CAPS")

            expected_fields = ['name', 'email', 'phone', 'skills', 'experience', 'education', 'location', 'languages', 'objectives']
            for field in expected_fields:
                if field not in parsed_data:
                    if field in ['skills', 'experience', 'education', 'languages']:
                        parsed_data[field] = []
                    else:
                        parsed_data[field] = ""

            if 'skills' in parsed_data and isinstance(parsed_data['skills'], list):
                hobbies = ['watching soccer', 'choir', 'soccer', 'football', 'singing', 'hobbies']
                original_skills = len(parsed_data['skills'])
                parsed_data['skills'] = [s for s in parsed_data['skills']
                                         if isinstance(s, str) and s.strip() and s.lower() not in hobbies]
                if len(parsed_data['skills']) < original_skills:
                    logger.info(f"   🧹 Removed {original_skills - len(parsed_data['skills'])} hobby items from skills")

            logger.info("   ✅ LLM parsing completed successfully")
            return parsed_data

        except json.JSONDecodeError as e:
            logger.error(f"   ❌ JSON parsing error: {str(e)}")

            repaired = _try_fix_truncated_json(output)
            if repaired is not None:
                logger.info("   ✨ Repaired truncated JSON successfully")
                return repaired

            return {
                "error": "JSON parsing failed",
                "error_details": str(e),
                "raw_extraction": output[:500]
            }

    except Exception as e:
        logger.error(f"   ❌ LLM inference failed: {str(e)}", exc_info=True)
        return {"error": f"LLM inference failed: {str(e)}"}

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    ---
    tags:
      - Health
    responses:
      200:
        description: Service is healthy
        schema:
          properties:
            status:
              type: string
              example: "healthy"
            model_loaded:
              type: boolean
              example: true
            model_path:
              type: string
              example: "/app/resume_parser/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
            model_size_mb:
              type: number
              example: 1065.6
            message:
              type: string
              example: "Resume parser microservice is running"
    """
    return jsonify({
        'status': 'healthy' if MODEL_LOADED else 'degraded',
        'model_loaded': MODEL_LOADED,
        'model_path': MODEL_PATH,
        'model_size_mb': round(os.path.getsize(MODEL_PATH) / (1024 * 1024), 1) if os.path.exists(MODEL_PATH) else 0,
        'message': 'Resume parser microservice is running'
    })

@app.route('/parse', methods=['POST'])
def parse_resume():
    """
    Parse a resume file (PDF, DOCX, TXT, DOC)
    ---
    tags:
      - Parse
    consumes:
      - multipart/form-data
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: Resume file (PDF, DOCX, TXT, or DOC)
    responses:
      200:
        description: Resume parsed successfully
        schema:
          properties:
            success:
              type: boolean
              example: true
            filename:
              type: string
              example: "resume.pdf"
            file_type:
              type: string
              example: ".pdf"
            data:
              type: object
              properties:
                name:
                  type: string
                  example: "JOHN DOE"
                email:
                  type: string
                  example: "john@example.com"
                phone:
                  type: string
                  example: "+1-555-0123"
                skills:
                  type: array
                  items:
                    type: string
                  example: ["Python", "Docker", "Node.js"]
                experience:
                  type: array
                  items:
                    type: object
                  example: [{"title": "Engineer", "company": "TechCorp", "duration": "2021-Present"}]
                education:
                  type: array
                  items:
                    type: object
                  example: [{"qualification": "B.S. CS", "institution": "University", "year": "2020"}]
            text_length:
              type: integer
              example: 2500
      400:
        description: Invalid request (no file or unsupported format)
      503:
        description: Model not loaded
    """
    if not MODEL_LOADED:
        logger.error("Parse request received but model not loaded")
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 503

    # Start overall timing
    overall_start = time.time()
    logger.info("=" * 60)
    logger.info("📋 NEW CV PARSING REQUEST STARTED")
    logger.info("=" * 60)

    if 'file' not in request.files:
        logger.warning("Parse request received without file")
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        logger.warning("Parse request received with empty filename")
        return jsonify({'error': 'Empty filename'}), 400

    logger.info(f"📄 Filename: {file.filename}")
    logger.info(f"📊 File size: {len(file.read())} bytes")
    file.seek(0)  # Reset file pointer

    # Check file extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in app.config['UPLOAD_EXTENSIONS']:
        logger.error(f"Unsupported file type: {ext}")
        return jsonify({
            'error': f'Unsupported file type. Supported: {app.config["UPLOAD_EXTENSIONS"]}'
        }), 400

    # Save file temporarily
    filename = secure_filename(file.filename)
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, filename)

    try:
        logger.info(f"💾 Saving file to: {file_path}")
        file.save(file_path)
        logger.info("✅ File saved successfully")

        # Extract text from file
        extract_start = time.time()
        logger.info("🔍 Step 1: Extracting text from file...")
        raw_text = extract_text(file_path)
        extract_time = time.time() - extract_start
        
        if not raw_text:
            logger.error("❌ Failed to extract text from file")
            return jsonify({'error': 'Could not extract text from file'}), 400
        
        logger.info(f"✅ Text extraction complete in {extract_time:.2f}s")
        logger.info(f"   Raw text length: {len(raw_text)} characters")

        # Clean the extracted text
        clean_start = time.time()
        logger.info("🧹 Step 2: Cleaning and normalizing text...")
        cleaned_text = clean_text(raw_text)
        clean_time = time.time() - clean_start
        
        logger.info(f"✅ Text cleaning complete in {clean_time:.2f}s")
        logger.info(f"   Cleaned text length: {len(cleaned_text)} characters")

        # Parse with local LLM
        llm_start = time.time()
        logger.info("🤖 Step 3: Processing with LLM (Qwen 2.5)...")
        parsed_data = parse_with_llm(cleaned_text)
        llm_time = time.time() - llm_start
        
        logger.info(f"✅ LLM processing complete in {llm_time:.2f}s")
        
        if 'error' in parsed_data:
            logger.warning(f"⚠️  LLM returned error: {parsed_data.get('error')}")
        else:
            logger.info(f"   ✓ Name: {parsed_data.get('name', 'N/A')}")
            logger.info(f"   ✓ Email: {parsed_data.get('email', 'N/A')}")
            logger.info(f"   ✓ Phone: {parsed_data.get('phone', 'N/A')}")
            logger.info(f"   ✓ Skills found: {len(parsed_data.get('skills', []))}")
            logger.info(f"   ✓ Experience entries: {len(parsed_data.get('experience', []))}")
            logger.info(f"   ✓ Education entries: {len(parsed_data.get('education', []))}")

        # Add metadata
        overall_time = time.time() - overall_start
        response = {
            'success': True,
            'filename': filename,
            'file_type': ext,
            'data': parsed_data,
            'text_length': len(cleaned_text),
            'processing_times': {
                'extraction_seconds': round(extract_time, 2),
                'cleaning_seconds': round(clean_time, 2),
                'llm_processing_seconds': round(llm_time, 2),
                'total_seconds': round(overall_time, 2)
            }
        }

        # Log completion summary
        logger.info("=" * 60)
        logger.info("✅ CV PARSING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"⏱️  TIMING SUMMARY:")
        logger.info(f"   • Text Extraction: {extract_time:.2f}s")
        logger.info(f"   • Text Cleaning:   {clean_time:.2f}s")
        logger.info(f"   • LLM Processing:  {llm_time:.2f}s")
        logger.info(f"   • TOTAL TIME:      {overall_time:.2f}s")
        logger.info("=" * 60)

        return jsonify(response)

    except Exception as e:
        overall_time = time.time() - overall_start
        logger.error(f"❌ Error during CV processing: {str(e)}", exc_info=True)
        logger.error(f"   Time elapsed before error: {overall_time:.2f}s")
        return jsonify({'error': str(e)}), 500

    finally:
        # Clean up temporary files
        try:
            logger.info("🧹 Cleaning up temporary files...")
            os.remove(file_path)
            os.rmdir(temp_dir)
            logger.info("✅ Temporary files cleaned up")
        except Exception as cleanup_error:
            logger.warning(f"⚠️  Could not clean up temporary files: {cleanup_error}")

@app.route('/parse-text', methods=['POST'])
def parse_text():
    """
    Parse raw text content
    ---
    tags:
      - Parse
    consumes:
      - application/json
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            text:
              type: string
              description: Resume text to parse
              example: "John Doe, Senior Software Engineer with 5 years experience in Python, Docker, and AWS..."
    responses:
      200:
        description: Text parsed successfully
        schema:
          properties:
            success:
              type: boolean
              example: true
            data:
              type: object
              properties:
                name:
                  type: string
                email:
                  type: string
                phone:
                  type: string
                skills:
                  type: array
                  items:
                    type: string
                experience:
                  type: array
                education:
                  type: array
            text_length:
              type: integer
      400:
        description: Invalid request (no text provided)
      503:
        description: Model not loaded
    """
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
    """
    Debug endpoint to check system information
    ---
    tags:
      - Debug
    responses:
      200:
        description: System debug information
        schema:
          properties:
            python_version:
              type: string
              example: "3.11.4 (main, Jun 7 2023..."
            platform:
              type: string
              example: "Linux-5.15.0-generic-x86_64"
            current_dir:
              type: string
              example: "/app"
            model_path:
              type: string
              example: "/app/resume_parser/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
            model_exists:
              type: boolean
              example: true
            model_loaded:
              type: boolean
              example: true
            files_in_dir:
              type: array
              items:
                type: string
              example: ["Dockerfile", "requirements.txt", "resume_microservice.py"]
    """
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
        print("\n" + "❌"*30)
        print("⚠️  STARTUP FAILED: Model could not be loaded!")
        print("❌"*30)
        print("\nThe service will NOT start accepting requests.")
        print("Check the debug endpoint at http://localhost:7860/debug")
        print("Common issues:")
        print("   - Insufficient RAM (need at least 4GB free)")
        print("   - Corrupted model file")
        print("   - Missing C++ runtime libraries")
        print("   - Network error during model download")
    else:
        print("\n" + "="*60)
        print("✅ ALL STARTUP CHECKS PASSED!")
        print("="*60)
        print("✅ Server ready to accept requests!")
        print("\n📡 Service running on port 7860")
        print("📝 Available endpoints:")
        print("   - GET  /health     - Health check")
        print("   - GET  /debug      - Debug information")
        print("   - GET  /apidocs/   - Interactive API documentation")
        print("   - POST /parse      - Upload resume file (PDF/DOCX/TXT)")
        print("   - POST /parse-text - Send raw text")
        print("\n📊 Model Information:")
        print(f"   - Model: Qwen 2.5 0.5B Instruct Fast (Quantized)")
        print(f"   - Size: {os.path.getsize(MODEL_PATH) / (1024 * 1024):.1f} MB")
        print(f"   - Path: {MODEL_PATH}")
        print("\n🌐 Access:")
        print("   - https://simbakm-resume-parser.hf.space")
        print("   - https://simbakm-resume-parser.hf.space/apidocs/")
        print("\n" + "="*60)

    app.run(host='0.0.0.0', port=7860, debug=False)