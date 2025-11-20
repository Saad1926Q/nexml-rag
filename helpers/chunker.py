import uuid
from typing import List, Dict

def simple_chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[Dict]:
    """
    Simple character-based chunker with overlap.
    Recommend: replace with token-aware chunker (tiktoken) if using tokens.
    Returns list of chunks with chunk_id, text, offset.
    """
    chunks = []
    i = 0
    text_len = len(text)
    while i < text_len:
        end = i + chunk_size
        chunk_text = text[i:end]
        chunks.append({
            "chunk_id": str(uuid.uuid4()),
            "text": chunk_text,
            "offset": i
        })
        i = end - overlap  
        if i < 0:
            i = 0
    return chunks
