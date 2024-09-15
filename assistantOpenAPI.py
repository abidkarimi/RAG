
# import os
# from openai import OpenAI
# import time
# from dotenv import load_dotenv
# load_dotenv()


# api_key = os.getenv('OPENAI_API_KEY')
# assistant_id = os.getenv('OPENAI_ASSISTANT_ID')
# thread_id = os.getenv('THREAD_ID')


# client = OpenAI(api_key=api_key)

# def save_uploaded_file(uploaded_file):
#     print("Saving")
#     with open(uploaded_file.name, "wb") as f:
#         f.write(uploaded_file.getbuffer())
        
       

# def get_assistant_response(query):
#     print("Query ", query)
#     threadId = thread_id
#     assistantId = assistant_id

#     message = client.beta.threads.messages.create(
#     thread_id=threadId,
#     role="user",
#     content=query
#         )
    
#     run = client.beta.threads.runs.create(
#     thread_id=threadId,
#     assistant_id=assistantId,
#     instructions="Use the user uploaded file to provide answer. Do not answer if you couldn't find any context in the knowledgebase. Just say I don't know"
#     )
    
#     while True:

#         ###### Check Run Status ######
#         run = client.beta.threads.runs.retrieve(thread_id=threadId,run_id=run.id)
#         time.sleep(1)
        
#         ###### Perform Actions based on run status #####
#         ###### If Run status is 'completed' it means return the recent output #####
#         if run.status == 'completed':
#             messages_ = client.beta.threads.messages.list(thread_id=threadId)
#             response = messages_.data[0].content[0].text.value
#             return response
        
       

# def update_assistant_knowledgebase(filepath):
#     print("Updating knowledgedbased")
#     my_assistant = client.beta.assistants.retrieve(assistant_id)
#     print("Up ", my_assistant)
#     file_ids = my_assistant.file_ids

#     # file = client.files.create(
#     # file=open(filepath, "rb"),
#     # purpose='assistants'
#     # )

#     # file_ids.append(file.id)

#     # my_updated_assistant = client.beta.assistants.update(
#     # assistant_id,
#     # tools=[{"type": "retrieval"}],
#     # file_ids=file_ids,
#     # )
#     print("assistant_knowledgebase_updated")
    

import os
import openai  # Correct import
import time
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
assistant_id = os.getenv('OPENAI_ASSISTANT_ID')
thread_id = os.getenv('THREAD_ID')

# Set the API key for openai
openai.api_key = api_key

def save_uploaded_file(uploaded_file):
    print("Saving")
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())

def get_assistant_response(query):
    print("Query ", query)
    
    # Call the OpenAI API directly
    try:
        # Send message to OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or another model like "gpt-4" if you have access
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{query}. Please provide the answer in markdown format with proper headings, numbered lists, and bullet points where appropriate."}
            ]
        )

        # Extract and return the assistant's response
        return response

    except Exception as e:
        print(f"Error occurred: {e}")
        return "An error occurred while processing your request. {e}"
    

def update_assistant_knowledgebase(filepath):
    print("Updating knowledge base")
    # OpenAI currently doesn't have a public API for file-based assistant knowledgebases,
    # but you can use files to upload knowledgebase-related information for other purposes.
    print("assistant_knowledgebase_updated")
