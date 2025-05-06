from collections import deque
from openai import OpenAI
from utils import search_kbase

def get_llm_response(prompt: str, client: OpenAI) -> str:
    """
    Calls OpenAI to get a response to the assembled prompt.
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def chat_loop(client: OpenAI, entries, index, k: int = 5):
    """
    Runs an interactive chat loop with the user, using vector‐search 
    to fetch relevant PDF chunks and then answering over them.
    Keeps the last 8 (user,agent) turns in memory.
    """
    # Chat history length
    history = deque(maxlen=8)

    print("--- Chat Started ---")
    print("Type your question (or 'exit' to quit):")

    while True:
        query = input("You: ").strip()
        if query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        # 1) Vector‐search over PDF chunks
        hits = search_kbase(query, client, entries, index, k=k)

        # 2) Build the PDF context block
        context_blocks = []
        for fn, sec, score in hits:
            excerpt = sec.text.replace("\n", " ")
            context_blocks.append(f"---\nFile: {fn}\nScore: {score:.3f}\n{excerpt}")
        context = "\n\n".join(context_blocks)

        # 3) Serialize the last 8 turns
        if history:
            hist_section = "\n".join(
                f"User: {u}\nAgent: {a}"
                for u, a in history
            ) + "\n\n"
        else:
            hist_section = ""

        # 4) Assemble the final prompt
        prompt = (
            "#You are a knowledgeable assistant who reads documents and answers questions.\n\n"
            f"{hist_section}"
            "# Use the following excerpts from the user's PDFs and prior chat history to answer the question as accurately and concisely as possible.\n\n"
            f"{context}\n\n"
            f"Question: {query}\nAnswer:"
        )

        # 5) Generate response to initial input
        answer = get_llm_response(prompt, client)
        print(f"\nAgent: {answer}\n")

        # 6) Append this dialogue turn to history
        history.append((query, answer))