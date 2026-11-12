
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2"


def ask_llm(prompt):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=180
    )

    response.raise_for_status()
    return response.json()["response"]


# ZERO-SHOT PROMPT
zero_shot_prompt = """
Classify the following HTTP status code into one category:
Success, Client Error, or Server Error.

HTTP Status Code: 503

Return only the category.
"""


# FEW-SHOT PROMPT

few_shot_prompt = """
You are an HTTP status code classifier.

Rules:
HTTP code	Expected category
201	        Success
401	        Client Error
403	        Client Error
503	        Server Error

Examples:

HTTP Status Code: 200
Category: Success

HTTP Status Code: 400
Category: Client Error

HTTP Status Code: 404
Category: Client Error

HTTP Status Code: 500
Category: Server Error

Now classify:

HTTP Status Code: 503

Return only one category.
"""


if __name__ == "__main__":

    print("\n--- ZERO-SHOT RESULT ---")
    print(ask_llm(zero_shot_prompt))

    print("\n--- FEW-SHOT RESULT ---")
    print(ask_llm(few_shot_prompt))
