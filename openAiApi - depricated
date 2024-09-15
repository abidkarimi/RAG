from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import re
import os
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Now you can access the OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')

# Initialize Sentence Transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Define paths
pdf_path = "docs/lec11_removed.pdf"
faiss_index_path = "faiss_index.index"

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    print("Extracting text from PDF")
    pdf_reader = PdfReader(pdf_path)
    pdf_text = ""
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()
    return pdf_text

# Function to chunk text
def chunk_text():
    print("Chunking text")
    text = extract_text_from_pdf(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to refine chunks
def refine_chunk(chunk):
    print("Refining chunk")
    chunk = re.sub(r'(?<!\S)(\S) (?=\S)', r'\1', chunk)
    chunk = re.sub(r'\s+', ' ', chunk.strip())
    return chunk

def refine_and_filter_chunks(chunks):
    print("Refining and filtering chunks")
    refined_chunks = [refine_chunk(chunk) for chunk in chunks]
    refined_chunks = [chunk for chunk in refined_chunks if len(chunk) > 20]
    return refined_chunks

# Function to create embeddings
def create_embeddings(chunks):
    print("Creating embeddings")
    embeddings = model.encode(chunks)
    return embeddings

# Function to store embeddings in FAISS
def store_embeddings_in_faiss(embeddings, index_path):
    print("Storing embeddings in FAISS")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to {index_path}")
    return index

# Function to load FAISS index from disk
def load_faiss_index(index_path):
    if os.path.exists(index_path):
        print(f"Loading FAISS index from {index_path}")
        index = faiss.read_index(index_path)
        return index
    else:
        print("FAISS index not found.")
        return None

# Function to perform semantic search
def search_query(query, index, model, chunks, top_k=2):
    print("Searching query:", query)
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    results = [chunks[i] for i in indices[0]]
    return results

# Function to call OpenAI for generating answers using RAG
def generate_answer(query, index, refined_chunks):
    print("Generating answer using RAG")
    llm = OpenAI(model_name="text-davinci-003", openai_api_key=openai_api_key)
    vectorstore = FAISS(model=model, index=index, texts=refined_chunks)
    retriever = RetrievalQA(llm=llm, retriever=vectorstore.as_retriever())
    answer = retriever(query)
    return answer['result']

# Main function
def callerFun():
    print("Caller function")
    index = load_faiss_index(faiss_index_path)
    
    if index is None:
        print("Index not found, generating new embeddings and creating a new index.")
        cleaned_chunks = chunk_text()
        refined_chunks = refine_and_filter_chunks(cleaned_chunks)
        embeddings = create_embeddings(refined_chunks)
        index = store_embeddings_in_faiss(embeddings, faiss_index_path)
    else:
        print("Index loaded successfully.")
        refined_chunks = refine_and_filter_chunks(chunk_text())
    
    query = "What are difference between Threads and Process"
    results = search_query(query, index, model, refined_chunks)

    # Generate an answer using RAG
    answer = generate_answer(query, index, refined_chunks)

    print(f"Search Results: {results}")
    print(f"Generated Answer: {answer}")

    return {
        'Result': results,
        'GeneratedAnswer': answer,
        'Query': query
    }
