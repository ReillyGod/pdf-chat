import json
from openai import OpenAI
from typing import List, Dict
from utils import ParsedDocs, find_new_papers, search_kbase

def rephrase_question(query: str, messages: List[Dict[str, str]], client: OpenAI) -> str:
    """
    Calls OpenAI to get a response to the assembled prompt.
    """
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
    
    resp = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=rephrase_prompt,
        max_tokens=512,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

def get_llm_response_functions(messages: List[Dict[str, str]], client: OpenAI, kbase: List[ParsedDocs]) -> List[Dict[str, str]]:
    """
    Calls OpenAI with the new tools API to support function-calling,
    and executes get_new_papers when requested.
    """
    # Function Definitions
    tools = [{
        "type": "function",
        "name": "get_new_papers",
        "description": "Fetch new PDFs from arXiv based on a search request. Only call this tool if the user asks for new papers on a topic specifically.",
        "parameters": {
            "type": "object",
            "properties": {
                "search_request": {
                    "type": "string",
                    "description": "Keywords to search with, keep it as concise as possible while still being spesific."
                },
                "top_n": {
                    "type": "integer",
                    "description": "Number of papers to fetch. If undefined assume 3"
                }
            },
            "required": ["search_request", "top_n"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "search_kbase",
        "description": "Search the Knowledgebase for current papers to answer the user's question. Unless they ask for a new paper, call this funciton.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_question": {
                    "type": "string",
                    "description": "The question that the user has asked about the existing papers. Simply repeat the user's question here."
                }
            },
            "required": ["user_question"],
            "additionalProperties": False
        },
        "strict": True
    }]

    paper_titles = []
    for paper in kbase:
        paper_titles.append(paper.filename)

    system_prompt = [{
        "role": "system",
        "content": ("# You are a knowledgeable assistant who reads user-provided PDF excerpts and answers questions as accurately and concisely as possible.\n"
                    "# Answer questions using the provided information from existing papers.\n"
                    "# Call the 'search_kbase' function whenever the user asks a question about an existing paper.\n"
                    "# You should almost allways call the 'search_kbase' function.\n"
                    "# If the user spesifically asks for new papers call the 'get_new_papers' function.\n"
                    "# When searching for new papers only include keywords, keep the input concise for best results.\n"
                    f"# Current list of papers: {paper_titles}"
        )
        }]

    response = client.responses.create(
        model="gpt-4o-mini-2024-07-18",
        input=system_prompt + messages,
        tools=tools,
    )

    tool_call = response.output[0]

    # Handle Function Calls
    if tool_call.type == "function_call":
        args = json.loads(tool_call.arguments)
        result = None

        if tool_call.name == "get_new_papers":
            print("Function Called: Get New Papers")

            # Call function
            result = find_new_papers(
                search_request=args["search_request"],
                top_n=args["top_n"]
            )

            messages.append(tool_call)
            messages.append({
                "type": "function_call_output",
                "call_id": tool_call.call_id,
                "output": f"Papers Found: {result}"
            })

            # Respond with results
            second_response = client.responses.create(
                model="gpt-4o-mini-2024-07-18",
                input=system_prompt + messages,
                tools=tools,
            )

            messages.append({"role": "assistant", "content": second_response.output_text})
            answer = messages[-1]["content"]
            print(f"\nAgent: {answer}\n")
            print("--- Please Reload Chat to Process New Papers ---\n")
            return messages

        elif tool_call.name == "search_kbase":
            print("Function Called: Search Kbase")

            # Call function
            result = search_kbase(
                query=args["user_question"],
                client=client,
                kbase=kbase
            )
            
            messages.append(tool_call)
            messages.append({
                "type": "function_call_output",
                "call_id": tool_call.call_id,
                "output": (f"# Use the context from the following papers to answer the user's question:\n{result}\n"
                           "# Keep your answers concise. Do not give long winded answers.\n"
                           "# Only answer the question with information provided by the paper.\n")
            })

            # Respond with results
            second_response = client.responses.create(
                model="gpt-4o-mini-2024-07-18",
                input=system_prompt + messages,
                tools=tools,
            )

            messages.append({"role": "assistant", "content": second_response.output_text})
            answer = messages[-1]["content"]
            print(f"\nAgent: {answer}\n")
            return messages
    

    # No function call, just return normal assistant message
    messages.append({"role": "assistant", "content": response.output_text})
    answer = messages[-1]["content"]
    print(f"\nAgent: {answer}\n")
    return messages

def chat_loop(client: OpenAI, kbase: List[ParsedDocs]):
    """
    Runs an interactive chat loop with the user, using vector‐search 
    to fetch relevant PDF chunks and then answering over them.
    Keeps the last 8 (user,assistant) turns in memory.
    """

    print("--- Chat Started ---")

    print("Current Papers:")
    for paper in kbase:
        print(paper.filename)
        
    print("Type your question (or 'e' to exit the chat):")

    messages = []


    # Main chat loop
    while True:
        # Each turn of dialogue has a ~4 messages, remove 4 messages if chat gets long
        if len(messages) >= 16:
            messages.pop(0)
            messages.pop(0)
            messages.pop(0)
            messages.pop(0)

        query = input("You: ").strip()
        if query.lower() in ("e", "q"):
            print("Goodbye!")
            break

        # Rephrase question to make sense in the context of the chat
        if len(messages) > 0:
            query = rephrase_question(query = query, messages = messages, client = client)
            print(f"Question Rephrased: {query}")
        
        messages.append({"role": "user", "content": query})

        # Start the function calling pipeline
        messages = get_llm_response_functions(messages, client, kbase)