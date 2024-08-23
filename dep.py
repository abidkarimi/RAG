import openai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

def generate_answer(query, context):
    prompt = f"Context:\n{context}\n\nQuery: {query}\n\nProvide a concise answer based on the context."

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use the chat model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,  # Adjust based on needs
        temperature=0.7  # Adjust creativity of responses
    )

    return response.choices[0].message['content'].strip()

# Example usage
query = "What are the differences between Threads and Processes?"
context = """Thread vs. Process: A process is an instance of a program that runs in its own memory space and context, while a thread is a lightweight unit of execution within a process that shares the same memory space."""

answer = generate_answer(query, context)
print(answer)
