
import os,uuid,json,PyPDF2
import numpy as np
from openai import OpenAI
import time
from dotenv import load_dotenv
load_dotenv()


api_key = os.getenv('OPENAI_API_KEY')
assistant_id = os.getenv('OPENAI_ASSISTANT_ID')
thread_id = os.getenv('THREAD_ID')


client = OpenAI(api_key=api_key)

def save_uploaded_file(uploaded_file):
    print("Saving")
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
       

def get_assistant_response(query):
    print("Query ", query)
    threadId = thread_id
    assistantId = assistant_id

    message = client.beta.threads.messages.create(
    thread_id=threadId,
    role="user",
    content=query
        )
    
    run = client.beta.threads.runs.create(
    thread_id=threadId,
    assistant_id=assistantId,
    instructions="Use the user uploaded file to provide answer. Do not answer if you couldn't find any context in the knowledgebase. Just say I don't know"
    )
    
    while True:

        ###### Check Run Status ######
        run = client.beta.threads.runs.retrieve(thread_id=threadId,run_id=run.id)
        time.sleep(1)
        
        ###### Perform Actions based on run status #####
        ###### If Run status is 'completed' it means return the recent output #####
        if run.status == 'completed':
            messages_ = client.beta.threads.messages.list(thread_id=threadId)
            response = messages_.data[0].content[0].text.value
            return {
                'generate_answer':response
                }
        
       

def update_assistant_knowledgebase(filepath):
    print("Updating knowledgedbased")
    my_assistant = client.beta.assistants.retrieve(assistant_id)
    print("Up ", my_assistant)
    file_ids = my_assistant.file_ids

    # file = client.files.create(
    # file=open(filepath, "rb"),
    # purpose='assistants'
    # )

    # file_ids.append(file.id)

    # my_updated_assistant = client.beta.assistants.update(
    # assistant_id,
    # tools=[{"type": "retrieval"}],
    # file_ids=file_ids,
    # )
    print("assistant_knowledgebase_updated")
    