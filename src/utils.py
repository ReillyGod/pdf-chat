from dotenv import load_dotenv
import os
import re
import tempfile
import json
import glob
import fitz
import base64
from dataclasses import dataclass
from typing import List

from openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
import numpy as np

load_dotenv()

@dataclass
class Section:
    text: str
    embedding: List[float]

class ParsedDocs:
    """
    Represents a parsed PDF document, its sections, and an overall document embedding.
    """
    def __init__(self, filename: str, sections: List[Section], document_embedding: List[float]):
        self.filename = filename
        self.sections = sections
        self.document_embedding = document_embedding

    def save_to_json(self, output_path: str) -> None:
        data = {
            "filename": self.filename,
            "document_embedding": self.document_embedding,
            "sections": [s.__dict__ for s in self.sections],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Saved JSON to {output_path}")

    @classmethod
    def load_from_json(cls, input_path: str) -> "ParsedDocs":
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        filename = data["filename"]
        document_embedding = data.get("document_embedding", [])
        sections = [Section(**sec) for sec in data.get("sections", [])]
        return cls(filename, sections, document_embedding)


def get_embedding(text: str, client: OpenAI, model: str = "text-embedding-3-small") -> List[float]:
    """
    Create an embedding for a text chunk using the given model.
    Assumes `text` is already within context length limits.
    """
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding


def get_page_description(base64_image: str, client: OpenAI) -> str:
    """Generate a description for a PDF page from a Base64‑encoded image."""
    prompt = (
        "# You are a document analyzer.\n"
        "# You will be given an image of a document and asked to write it out in text format.\n"
        "# For text in the document, repeat it back in its entirety word for word.\n"
        "# For a table write out the table and all of its figures.\n"
        "# For images and figures, give an extremely detailed text based description of the image.\n"
        "# It should be detailed enough that I can picture the image or figure correctly with my eyes closed.\n"
        "# Maintain the order of the text, tables, and images / figures as they appear in the document.\n"
        "# If there are multiple columns or any other odd formatting, output the text in the order a human would read it.\n"
        "# Only output the document text.\n"
        "# Do not include anything not present in the document.\n"
        "# Output the text in a readable markdown format.\n"
    )
    response = client.chat.completions.create(
        model="o4-mini-2025-04-16",
        messages=[
            {"role": "system",  "content": prompt},
            {"role": "user",    "content": [
                {"type": "text", "text": "Process this pdf page as instructed."},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
            ]},
        ],
    )
    output = response.choices[0].message.content
    print(f"Page Processed:{output[100:]}")
    return output


def process_pdf(pdf_path: str, client: OpenAI) -> ParsedDocs:
    """
    Process a PDF by:
      1. Rendering each page to an image
      2. Generating a markdown description per page via get_page_description()
      3. Stripping the ```markdown fences``` and concatenating all pages
      4. Splitting into ~2k‑char chunks (200 char overlap)
      5. Embedding each chunk
      6. Computing a single document embedding as the average of all section embeddings

    Returns a ParsedDocs instance.
    """
    filename = os.path.basename(pdf_path)
    print(f"Processing {filename}…")

    page_texts: List[str] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        doc = fitz.open(pdf_path)
        for pg in range(len(doc)):
            pix = doc.load_page(pg).get_pixmap(matrix=fitz.Matrix(2, 2))
            png_bytes = pix.tobytes("png")
            b64 = base64.b64encode(png_bytes).decode("utf-8")
            raw_md = get_page_description(b64, client)
            cleaned = re.sub(r"^```(?:markdown)?\s*", "", raw_md)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            page_texts.append(cleaned)

    full_text = "\n\n".join(page_texts)
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_text(full_text)

    parsed_secs: List[Section] = []
    for chunk in chunks:
        print(chunk[:100] + "…")
        emb = get_embedding(chunk, client)
        parsed_secs.append(Section(text=chunk, embedding=emb))

    # Compute document embedding
    all_embs = np.array([sec.embedding for sec in parsed_secs], dtype="float32")
    doc_emb = all_embs.mean(axis=0).tolist()

    return ParsedDocs(filename, parsed_secs, doc_emb)


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Return the cosine similarity between two vectors.
    """
    num = np.dot(emb1, emb2)
    denom = (np.linalg.norm(emb1) * np.linalg.norm(emb2)) + 1e-10
    return float(num / denom)


def build_kbase(json_folder: str) -> List[ParsedDocs]:
    """
    Load every JSON in `json_folder` as a ParsedDocs object and return the list.
    """
    docs: List[ParsedDocs] = []
    for path in glob.glob(os.path.join(json_folder, "*.json")):
        parsed = ParsedDocs.load_from_json(path)
        docs.append(parsed)
    return docs


def search_kbase(
    query: str,
    client: OpenAI,
    kbase: List[ParsedDocs],
) -> str:
    """
    1) Embed `query` and score against each document.embedding.
       Keep up to the top‑3 documents within 0.05 of the best.
    2) For each selected document:
       • Compute cosine similarity of query to each section.embedding.
       • Pick the 3 highest‑scoring sections and their immediate neighbors.
    3) Print diagnostics and return a concatenated string of results.
    """
    # 1) Embed the query once
    q_emb = np.array(get_embedding(query, client), dtype="float32")
    q_emb /= (np.linalg.norm(q_emb) + 1e-10)

    # Document-level scoring
    doc_scores = []  # List of (ParsedDocs, score)
    for doc in kbase:
        d_emb = np.array(doc.document_embedding, dtype="float32")
        score = cosine_similarity(d_emb, q_emb)
        doc_scores.append((doc, score))
    # Sort descending by score
    doc_scores.sort(key=lambda x: x[1], reverse=True)

    if not doc_scores:
        print("No documents found in kbase.")
        return ""

    top_score = doc_scores[0][1]
    # Keep up to 3 docs within 0.05 of top_score
    selected = [(doc, score) for doc, score in doc_scores[:3] if score >= top_score - 0.1]
    sel_info = [f"{doc.filename} (score={score:.4f})" for doc, score in selected]
    print(f"Selected documents: {', '.join(sel_info)}")

    output_parts: List[str] = []
    # 2) Section-level selection
    for doc, _ in selected:
        sec_scores = np.array([
            cosine_similarity(np.array(sec.embedding, dtype="float32"), q_emb)
            for sec in doc.sections
        ])
        # Top-3 indices
        top_idxs = np.argsort(-sec_scores)[:3]

        # Include sections before and after each section
        # Ensures tables or image descriptions are fully intact
        pick_idxs = set()
        for idx in top_idxs:
            pick_idxs.add(idx)
            if idx - 1 >= 0:
                pick_idxs.add(idx - 1)
            if idx + 1 < len(doc.sections):
                pick_idxs.add(idx + 1)
        ordered = sorted(pick_idxs)
        print(f"Document '{doc.filename}': selecting {len(ordered)} sections")

        # Build output block for this document
        parts = [f"Document: {doc.filename}"]
        for i in ordered:
            parts.append(doc.sections[i].text.strip())
            parts.append("\n---\n")
        block = "\n".join(parts).rstrip("\n---\n")
        output_parts.append(block)

    # 3) Combine all document blocks
    return "\n\n".join(output_parts)