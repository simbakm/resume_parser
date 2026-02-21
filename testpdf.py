import os
import PyPDF2
import sys


def diagnose_pdf(file_path):
    """Diagnose PDF file issues"""

    print(f"\n🔍 Diagnosing PDF: {file_path}")
    print("=" * 60)

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False

    # Check file size
    file_size = os.path.getsize(file_path)
    print(f"📊 File size: {file_size} bytes ({file_size / 1024:.1f} KB)")

    if file_size == 0:
        print("❌ File is empty (0 bytes)")
        return False

    # Try to open and read the PDF
    try:
        with open(file_path, 'rb') as file:
            # Read first few bytes to check PDF signature
            header = file.read(5)
            if header.startswith(b'%PDF'):
                print(f"✅ Valid PDF header: {header}")
            else:
                print(f"❌ Invalid PDF header: {header}")
                print("   PDF files should start with '%PDF'")
                return False

            # Try to parse with PyPDF2
            file.seek(0)
            pdf_reader = PyPDF2.PdfReader(file)

            num_pages = len(pdf_reader.pages)
            print(f"✅ PyPDF2 can read file")
            print(f"📄 Number of pages: {num_pages}")

            # Try to extract text from first page
            if num_pages > 0:
                first_page = pdf_reader.pages[0]
                text = first_page.extract_text()
                if text and text.strip():
                    print(f"✅ Text extracted from page 1 ({len(text)} chars)")
                    print("\n📝 First 200 chars of extracted text:")
                    print("-" * 40)
                    print(text[:200])
                    print("-" * 40)
                else:
                    print("⚠️  No text extracted from page 1 (might be scanned/image-based)")

            return True

    except PyPDF2.errors.PdfReadError as e:
        print(f"❌ PyPDF2 cannot read PDF: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def try_alternative_extraction(file_path):
    """Try alternative PDF extraction methods"""

    print("\n🔄 Trying alternative extraction methods...")

    # Method 1: Try pdfplumber if available
    try:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

            if text.strip():
                print("✅ pdfplumber extracted text successfully")
                print(f"📝 Extracted {len(text)} characters")
                return text
    except ImportError:
        print("⚠️ pdfplumber not installed")
    except Exception as e:
        print(f"❌ pdfplumber failed: {e}")

    # Method 2: Try pypdf (different library)
    try:
        import pypdf
        with open(file_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

            if text.strip():
                print("✅ pypdf extracted text successfully")
                return text
    except ImportError:
        print("⚠️ pypdf not installed")
    except Exception as e:
        print(f"❌ pypdf failed: {e}")

    return None


if __name__ == "__main__":
    # Replace with your PDF path
    pdf_path = r"C:\Users\support\Downloads\Image_5.pdf"  # Update this path

    if diagnose_pdf(pdf_path):
        print("\n✅ PDF is readable by PyPDF2")
    else:
        print("\n⚠️ PDF has issues, trying alternatives...")
        extracted_text = try_alternative_extraction(pdf_path)

        if extracted_text:
            print("\n✅ Successfully extracted text with alternative method!")
            print("\n📝 Full extracted text:")
            print("=" * 60)
            print(extracted_text)
            print("=" * 60)

            # Save extracted text for manual review
            with open('extracted_text.txt', 'w', encoding='utf-8') as f:
                f.write(extracted_text)
            print("💾 Extracted text saved to extracted_text.txt")