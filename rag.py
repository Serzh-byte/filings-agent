"""Build and search the report index.

    python rag.py build [size overlap]
    python rag.py ask "question" [k]
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

DOCS_DIR = Path("docs")
INDEX_DIR = Path("index")


def chunk_text(text, size=800, overlap=100):
    assert size > overlap, "size must be bigger than overlap"
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


_model = None


def embed(texts):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("all-MiniLM-L6-v2")
    # normalised, so a dot product between two vectors is their cosine similarity
    return np.asarray(_model.encode(texts, normalize_embeddings=True))


def extract_pages(pdf_path, extractor="pymupdf"):
    # pypdf is only here for comparison, it drops spaces on the AEM report
    if extractor == "pymupdf":
        import pymupdf

        with pymupdf.open(str(pdf_path)) as doc:
            for n, page in enumerate(doc, start=1):
                yield n, page.get_text()
    else:
        from pypdf import PdfReader

        for n, page in enumerate(PdfReader(str(pdf_path)).pages, start=1):
            yield n, page.extract_text() or ""


def strip_repeated_lines(pages, min_share=0.6):
    """Drop lines (page headers, mostly) that appear on most pages."""
    if len(pages) < 5:
        return pages
    counts = {}
    for _, text in pages:
        for line in {ln.strip() for ln in text.splitlines() if ln.strip()}:
            counts[line] = counts.get(line, 0) + 1
    repeated = {ln for ln, c in counts.items() if c >= min_share * len(pages)}
    cleaned = []
    for n, text in pages:
        kept = [ln for ln in text.splitlines() if ln.strip() not in repeated]
        cleaned.append((n, "\n".join(kept)))
    return cleaned


def load_pdfs(folder=DOCS_DIR, size=800, overlap=100, extractor="pymupdf", strip_headers=True):
    records = []
    for pdf in sorted(Path(folder).glob("*.pdf")):
        company = pdf.stem.split("_")[0]  # AEM_q3_2026.pdf -> AEM
        pages = list(extract_pages(pdf, extractor))
        if strip_headers:
            pages = strip_repeated_lines(pages)
        for page_number, text in pages:
            for chunk in chunk_text(text, size, overlap):
                records.append({"doc": pdf.name, "company": company, "page": page_number, "text": chunk})
    return records


def build_index(records, embed_fn=embed, index_dir=INDEX_DIR):
    index_dir = Path(index_dir)
    index_dir.mkdir(exist_ok=True)
    vectors = embed_fn([r["text"] for r in records])
    np.save(index_dir / "vectors.npy", vectors)
    (index_dir / "records.json").write_text(json.dumps(records))
    return vectors


def load_index(index_dir=INDEX_DIR):
    index_dir = Path(index_dir)
    vectors = np.load(index_dir / "vectors.npy")
    records = json.loads((index_dir / "records.json").read_text())
    return vectors, records


def search(query, vectors, records, embed_fn=embed, k=4, company=None):
    q = embed_fn([query])[0]
    scores = vectors @ q

    if company:
        mask = np.array([r["company"].lower() == company.lower() for r in records])
        scores = np.where(mask, scores, -np.inf)

    best = np.argsort(scores)[::-1][:k]
    return [{**records[i], "score": float(scores[i])} for i in best if np.isfinite(scores[i])]


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "build":
        size = int(sys.argv[2]) if len(sys.argv) > 2 else 800
        overlap = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        recs = load_pdfs(size=size, overlap=overlap)
        build_index(recs)
        print(f"Indexed {len(recs)} chunks from {len({r['doc'] for r in recs})} documents")
    elif len(sys.argv) >= 3 and sys.argv[1] == "ask":
        k = int(sys.argv[3]) if len(sys.argv) > 3 else 4
        vecs, recs = load_index()
        for hit in search(sys.argv[2], vecs, recs, k=k):
            print(f"[{hit['score']:.2f}] {hit['doc']} p.{hit['page']}: {hit['text'][:400]}...\n")
    else:
        print(__doc__)