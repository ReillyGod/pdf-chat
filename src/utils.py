# Imports
from dotenv import load_dotenv
import os
import json
import glob
from dataclasses import dataclass
from typing import List, Dict, Tuple

from openai import OpenAI
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import numpy as np
import faiss

load_dotenv()

# Some simple classes to keep data stuff straight and extendable
@dataclass
class Section:
    text: str
    embedding: List[float]

class ParsedDocs:
    def __init__(self, docs: Dict[str, List[Section]] = None):
        self.docs = docs or {}

    def __getitem__(self, key):
        return self.docs[key]

    def __setitem__(self, key, value):
        self.docs[key] = value

    def items(self):
        return self.docs.items()

    def save_to_json(self, output_path: str) -> None:
        serializable = {f: [s.__dict__ for s in secs] for f, secs in self.docs.items()}
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2)
        print(f"Saved JSON to {output_path}")
        
    @classmethod
    def load_from_json(cls, input_path: str) -> "ParsedDocs":
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        docs = {
            fname: [Section(**sec) for sec in sec_list]
            for fname, sec_list in data.items()
        }
        return cls(docs)


def extract_chunks(pdf_path: str) -> List[str]:
    """
    Read the PDF, extract all text, and split into chunks of ~1000 characters
    with 200 characters overlap.
    """
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text

    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(text)


def get_embedding(text: str, client: OpenAI, model: str = "text-embedding-3-small") -> List[float]:
    """
    Create an embedding for a text chunk using the given model.
    Assumes `text` is already within context length limits.
    """
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding


def process_pdf(pdf_path: str, client: OpenAI) -> ParsedDocs:
    """
    Process a PDF by:
      1. Extracting text chunks
      2. Generating a tag and description for each chunk
      3. Embedding each chunk
    Returns a ParsedDocs instance.
    """
    filename = os.path.basename(pdf_path)
    print(f"Processing {filename}…")

    chunks = extract_chunks(pdf_path)
    parsed_secs: List[Section] = []

    for chunk in chunks:
        print(chunk[:100] + "…")

        embedding = get_embedding(chunk, client)
        parsed_secs.append(Section(text=chunk, embedding=embedding))

    return ParsedDocs({filename: parsed_secs})


def build_kbase(json_folder: str) -> Tuple[List[Tuple[str, Section]], faiss.IndexFlatIP]:
    """
    Scans `json_folder` for all .json files that were emitted by ParsedDocs.save_to_json,
    loads them, flattens all Sections into a single list, normalizes their embeddings,
    and builds a FAISS inner‐product index for cosine search.

    Returns:
      - entries: a list of (filename, Section) tuples in the same order as the index
      - index: a faiss.IndexFlatIP you can call .search() on
    """
    entries: List[Tuple[str, Section]] = []
    for path in glob.glob(os.path.join(json_folder, "*.json")):
        parsed = ParsedDocs.load_from_json(path)
        for fn, secs in parsed.items():
            for sec in secs:
                entries.append((fn, sec))

    embs = np.array([sec.embedding for _, sec in entries], dtype="float32")

    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / (norms + 1e-10)

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    return entries, index


def search_kbase(
    query: str,
    client: OpenAI,
    entries: List[Tuple[str, Section]],
    index: faiss.IndexFlatIP,
    k: int = 5
) -> List[Tuple[str, Section, float]]:
    """
    Given a user query:
      1. embed it with your existing `get_embedding`
      2. normalize for cosine
      3. search the FAISS index for top-k hits

    Returns a list of (filename, Section, score) sorted by descending similarity.
    """

    q_emb = np.array(get_embedding(query, client), dtype="float32")
    q_emb /= (np.linalg.norm(q_emb) + 1e-10)
    q_emb = q_emb.reshape(1, -1)

    scores, idxs = index.search(q_emb, k)

    results: List[Tuple[str, Section, float]] = []
    for score, idx in zip(scores[0], idxs[0]):
        fn, sec = entries[idx]
        results.append((fn, sec, float(score)))
    return results
