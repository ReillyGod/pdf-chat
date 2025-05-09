from openai import OpenAI
from utils import search_kbase
from typing import List, Dict
from utils import ParsedDocs

def get_llm_response(messages: List[Dict[str, str]], client: OpenAI) -> str:
    """
    Calls OpenAI to get a response to the assembled prompt.
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=messages,
        max_tokens=512,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

def get_llm_response(messages: List[Dict[str, str]], client: OpenAI) -> str:
    """
    Calls OpenAI to get a response to the assembled prompt.
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=messages,
        max_tokens=512,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def chat_loop(client: OpenAI, kbase: List[ParsedDocs]):
    """
    Runs an interactive chat loop with the user, using vector‐search 
    to fetch relevant PDF chunks and then answering over them.
    Keeps the last 8 (user,agent) turns in memory.
    """

    print("--- Chat Started ---")
    print("Type your question (or 'exit' to quit):")

    messages = []

    while True:
        if len(messages) >= 8:
            messages.pop(0)
            messages.pop(0)

        query = input("You: ").strip()
        if query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        # Rephrase question to make sense in the context of the chat
        if len(messages) > 0:
            rephrase_prompt = [{
                "role": "system",
                "content": ("# You are a linguist who likes to rephrase questions\n"
                            "# Given a chat history rephrase the current question so that it makes sense in the context of the chat.\n"
                            "# Ensure that the new question functions as a standalone question without the chat context.\n"
                            "```CHAT HISTORY\n"
                            f"{messages}\n"
                            "```\n"
                            "# Only respond with the rephrased question.\n"
                            "# If the new question references any named entities in the chat history, explicitly include this in the new question."
                            "# Do not include anything else in your response. Only respond with the individual question."
                )
                },
                {
                "role": "user",
                "content": query
                }]
            
            print(rephrase_prompt)
            
            # Use this as the new query for similarty search
            query = get_llm_response(rephrase_prompt, client)
            print(f"Question Rephrased: {query}")


        # 1) Vector‐search over PDF chunks
        context = search_kbase(query=query, client=client, kbase=kbase)

        system_prompt = [{
        "role": "system",
        "content": ("# You are a knowledgeable assistant who reads user-provided PDF excerpts and answers questions as accurately and concisely as possible.\n"
                    "# Answer questions using the provided information from papers.\n"
                    "```PDF DATA\n"
                    f"{context}\n"
                    "```"
        )
        }]
        
        messages.append({"role": "user", "content": query})

        # 5) Generate response to initial input
        answer = get_llm_response(system_prompt + messages, client)
        print(f"\nAgent: {answer}\n")

        # 6) Append this dialogue turn to history
        messages.append({"role": "assistant", "content": answer})