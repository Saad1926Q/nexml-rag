import io
from typing import Tuple
import fitz        
import docx
import mimetypes

def extract_text_from_file_content(file_bytes: bytes, filename: str) -> str:
    """
    Extract text from a PDF or DOCX file bytes.
    Returns the concatenated text.
    """
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return _extract_text_from_pdf_bytes(file_bytes)
    elif lower.endswith(".docx"):
        return _extract_text_from_docx_bytes(file_bytes)
    else:        
        mime, _ = mimetypes.guess_type(filename)
        if mime == "application/pdf":
            return _extract_text_from_pdf_bytes(file_bytes)
        raise ValueError("Unsupported file type. Only PDF and DOCX are supported.")

def _extract_text_from_pdf_bytes(b: bytes) -> str:
    text_parts = []
    with fitz.open(stream=b, filetype="pdf") as doc:
        for page in doc:
            text = page.get_text("text")
            if text:
                text_parts.append(text)
    return "\n\n".join(text_parts)

def _extract_text_from_docx_bytes(b: bytes) -> str:
    bio = io.BytesIO(b)
    doc = docx.Document(bio)
    paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n\n".join(paragraphs)
