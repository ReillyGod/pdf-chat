# PDF‐Chat

A simple tool that ingests PDF files, builds a vector search index over their text, and then lets you chat against your PDF “knowledge base” using OpenAI function‐calling.

## Features

- **PDF Processing**: Splits PDFs into overlapping text chunks, generates embeddings via OpenAI.
- **Vector Index**: Uses FAISS to index and search those embeddings.
- **CLI Chat Interface**: Interactive REPL that retrieves relevant chunks on demand each question.

## Installation

**Clone the github repo**

```bash
git clone https://github.com/reillygod/pdf-chat.git
cd pdf-chat
```

**Create and activate a Conda environment (or venv)**

```bash
conda create --name pdf-chat python=3.12 -y
conda activate pdf-chat
```

**Install Python dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

- Add the OpenAI API key to the .env file
- Add any PDFs to the pdf folder (a few have been included and pre processed)
- The PDF names are passed to the LLM so a descriptive name is generally better

## Usage

**Run the main file**
```bash
python src/main.py
```

- Once the PDF's have processed enter your question in the chat
- Type 'exit' in the chat or enter 'Ctrl+C' to close the chat

## Writeup

So the main goal here was to efficiently parse a PDF’s text. I omitted image and graph / table processing for this version due to time constraints, this would be the first thing I would address given more time. The PDF’s parsing was a bit tricky to get right and I tried using a few libraries to handle more complex objects. If the parsing was done I would use multi-modal models to generate descriptions of the images / data in the charts, adding them into the text corpus for each pdf.

Once the PDF has been converted into text, I am splitting it into 1000 token chunks with 200 tokens of overlap between chunks. The search is performed by embedding the user’s input and calculating the cosine similarity between the input and the chunk’s embedding. The chunk’s text is then retrieved and passed to the model as context to answer the question. For simplicity sake I am storing the chunks and embeddings in a folder called kbase (knowledgebase) so that they do not need to be processed at runtime every instance. Long term you would use a vector db like weaviate or pgvector on top of prostgres. 

The chat maintains an 8 turns of dialogue as chat history to provide some added context outside of a single turn of RAG. When asking comparisons between PDFs or compound questions to this version of the app, it is best to split them into separate fully formed questions. This is something that can be done automatically through question rephrasing and splitting. I implemented something similar at Seek, but again time constraints caused me to omit it from this demo.

Overall this is a pretty general approach for RAG over large text corpuses, but it can definitely be tweaked further for specific purposes. For example a multilevel chunking process where you create larger meta-chunks out of 10-20 individual chunks. This would help in dealing with longer individual documents as you can perform an initial search on the meta chunks to identify broad sections of text and then a secondary search on the individual chunks within them to get more specific details.

Thanks for taking the time to read through this!


