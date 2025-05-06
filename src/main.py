# Imports
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

from utils import process_pdf, build_kbase
from chat import chat_loop

def main():
    # Boring init stuff
    project_root = Path(__file__).parent.parent
    load_dotenv(project_root / ".env")

    openai_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=openai_key)

    # Ensure kbase/ exists
    kbase_dir = project_root / "kbase"
    kbase_dir.mkdir(exist_ok=True)

    # Define PDF input directory
    pdf_dir = project_root / "pdf"

    # Determine which PDFs have already been processed (by checking for .json files)
    all_pdfs = sorted(pdf_dir.glob("*.pdf"))
    processed_jsons = {p.stem for p in kbase_dir.glob("*.json")}
    to_process = [p for p in all_pdfs if p.stem not in processed_jsons]

    # Process PDFs that do not appear in the kbase folder
    if to_process:
        print(f"Found {len(to_process)} new PDF(s) to process...")
        for pdf_path in to_process:
            new_data = process_pdf(str(pdf_path), client)
            output_fp = kbase_dir / f"{pdf_path.stem}.json"
            new_data.save_to_json(str(output_fp))
            print(f"Saved parsed data for {pdf_path.name} to {output_fp.name}")
        print("PDF processing complete.")
    else:
        print("No new PDFs to process.")

    # Load the knowledgebase for RAG
    print("Building the knowledgebase...")
    entries, index = build_kbase(str(kbase_dir))
    print(f"Indexed {len(entries)} sections.")

    # Start the chat
    chat_loop(client, entries, index)

if __name__ == "__main__":
    main()
