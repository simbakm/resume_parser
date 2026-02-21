import requests
import os
import json


def parse_pdf_resume(file_path):
    """
    Parse a PDF resume file

    Args:
        file_path: Full path to the PDF file

    Returns:
        Parsed data as dictionary, or None if error
    """
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None

    try:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'application/pdf')}
            response = requests.post('http://localhost:5000/parse', files=files, timeout=60)

        if response.status_code == 200:
            return response.json().get('data', {})
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            return None

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


# === HOW TO USE ===
# Just change this path to your PDF file:
my_file = r"C:\Users\support\Documents\Munyaradzi Daniel Tamayi Resume1.pdf"
result = parse_pdf_resume(my_file)

if result:
    print("\n✅ Parsed successfully!")
    print(json.dumps(result, indent=2, ensure_ascii=False))