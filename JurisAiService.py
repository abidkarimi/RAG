from PyPDF2 import PdfReader       #func 1
from langchain.text_splitter import RecursiveCharacterTextSplitter  #func 2
from sentence_transformers import SentenceTransformer
import faiss
import re
import os
# from langchain_community.llms import OpenAI
import openai

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


#func 1
pdf_path = "./docs/100_pages.pdf"
faiss_index_path = "faiss_index.index"


def process_data():
    """
    A simple function to process input data.
    """
    # For demonstration, let's just return the input data in uppercase.
    return callerFun()

def extract_text_from_pdf(pdf_path):
    print("Extract Text from pdf", pdf_path)
    pdf_reader = PdfReader(pdf_path)
    pdf_text = ""
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()
    return pdf_text


#second function 
# Step 2: Chunk Text
def chunk_text():
    print("Chunk Text")
    text = extract_text_from_pdf(pdf_path="./docs/100_pages.pdf")
    #chunk size using langchain
    # Initialize the text splitter
    print("Chunking of file", pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,  # Set the desired chunk size in characters
        chunk_overlap=150,  # Set the desired overlap between chunks
    )

    # Split the text into chunks
    chunks = text_splitter.split_text(text) # text is now defined as an argument to the function
    
    return chunks #return the chunks

# Step 3: Refine Chunks
def refine_chunk(chunk):
    print("Refine Chunk")
    chunk = re.sub(r'(?<!\S)(\S) (?=\S)', r'\1', chunk)
    chunk = re.sub(r'\s+', ' ', chunk.strip())
    return chunk

def refine_and_filter_chunks(chunks):
    print("Refine and fiter chunk")
    refined_chunks = [refine_chunk(chunk) for chunk in chunks]
    refined_chunks = [chunk for chunk in refined_chunks if len(chunk) > 20]
    return refined_chunks

# cleaned_chunks = chunk_text(extract_text_from_pdf())
def create_embeddings(chunks, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    print("Create Embeddings")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    return embeddings


def store_embeddings_in_faiss(embeddings, index_path):
    print("Store Embeddings In Faiss")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Save the FAISS index to disk
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to {index_path}")
    return index

def load_faiss_index(index_path):
    if os.path.exists(index_path):
        print(f"Loading FAISS index from {index_path}")
        index = faiss.read_index(index_path)
        return index
    else:
        print("FAISS index not found.")
        return None


# Step 6: Implement Semantic Search
def search_query(query, index, model, chunks, top_k=2):
    print("Search Query", query)
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    results = [chunks[i] for i in indices[0]]
    return results

# Function to call OpenAI for generating answers using RAG
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

def callerFun(query_from_user):
    print("Caller Fun")
    
    # Load the FAISS index from disk if it exists
    index = load_faiss_index(faiss_index_path)
    
    # Check if the index was successfully loaded
    if index is None:
        print("Index not found, generating new embeddings and creating a new index.")
        
        # Process and create new embeddings if index not found
        cleaned_chunks = chunk_text()
        refined_chunks = refine_and_filter_chunks(cleaned_chunks)
        embeddings = create_embeddings(refined_chunks)
        
        # Store FAISS index on disk
        index = store_embeddings_in_faiss(embeddings, faiss_index_path)
    else:
        print("Index loaded successfully.")
        
        # If index is loaded, you still need the refined chunks for searching
        refined_chunks = refine_and_filter_chunks(chunk_text())
    
    # Example Semantic Search
    query = query_from_user
    results = search_query(query, index, model, refined_chunks)

    for i, result in enumerate(results):
        print(f"Result {i+1}: {result}")

    return {
        # 'Result': results,
        # 'Query': query,
        # 'Keu': openai_api_key,
        'generate_answer': generate_answer(query, results)
    }
