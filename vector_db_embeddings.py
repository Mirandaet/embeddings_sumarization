from transformers import AutoTokenizer, AutoModel
import torch
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection, connections, utility
import os
import requests
import json
import pickle
from pymilvus import Index
from openai import OpenAI
import numpy as np
import ast

# Initialize tokenizer and model from HuggingFace Transformers
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

id_to_filename = {}


# Generate embeddings from code snippets
def generate_embeddings(code_snippets, model="text-embedding-3-large", max_length=2048):
    embeddings_with_filenames = []
    for filename, snippet in code_snippets:
        # Split the snippet into chunks
        chunks = extract_code_blocks(snippet)
        
        # Store all embeddings for the current file
        file_embeddings = []

        for chunk in chunks:
            try:
                response = client.embeddings.create(
                    input=chunk,
                    model=model
                )
                embedding = response.data[0].embedding
                file_embeddings.append(embedding)
            except Exception as e:
                print(f"Failed to generate embedding for chunk in {filename}: {str(e)}")
                file_embeddings.append(None)  # Use None or a default embedding vector for error handling

        # Aggregating embeddings (e.g., by averaging)
        if file_embeddings:
            valid_embeddings = [emb for emb in file_embeddings if emb is not None]
            if valid_embeddings:
                # Average the embeddings
                averaged_embedding = np.mean(valid_embeddings, axis=0)
                embeddings_with_filenames.append((filename, averaged_embedding))
            else:
                embeddings_with_filenames.append((filename, None))

    if embeddings_with_filenames:
        print("Embeddings generated for each file.")

    # Save embeddings with filenames
    with open('embeddings_with_filenames.pkl', 'wb') as f:
        pickle.dump(embeddings_with_filenames, f)

    return embeddings_with_filenames


def create_collection(embedding_dimension):
    connections.connect(alias="default", host='localhost', port='19530')
    collection_name = "openAI_embeddings"

    if utility.has_collection(collection_name):
        collection = Collection(name=collection_name)
        collection.drop()
        print(f"Dropped existing collection: {collection_name}")
        
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dimension),
        FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=255)
    ]
    schema = CollectionSchema(fields, description="Embeddings with filenames")
    collection = Collection(name=collection_name, schema=schema)

    return collection


def extract_code_snippets(directory):
    print("code extraction")
    code_snippets = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file) 
                try:
                    with open(os.path.join(root, file), 'r') as f:
                        code = f.read()
                        if code:  # Check if the file is not empty
                            code_snippets.append((file_path, code))
                except IOError as e:
                    print(f"Error reading file {file}: {e}")
    return code_snippets




# Insert embeddings into Milvus
def insert_embeddings(code_snippets, embeddings_file, collection, starter_embeddings = None):

    # Load or generate embeddings
    if not os.path.exists(embeddings_file):
        if starter_embeddings == None:
            embeddings = generate_embeddings(code_snippets)
            save_embeddings(embeddings, embeddings_file)
        else:
            save_embeddings(starter_embeddings, embeddings_file)
    else:
        print("")
        embeddings = load_embeddings(embeddings_file)

    if not embeddings:
        print("No embeddings generated or loaded.")
        return

    # Prepare documents for insertion, filter out None values or use a default vector
    documents = []
    for snippet, emb in zip(code_snippets, embeddings):
        if emb is None:
            emb = [0]*768  # Default vector if embedding generation failed
        documents.append({"embedding": emb[1], "filename": snippet[0]})

    # Insert documents into Milvus
    if documents:
        mr = collection.insert(documents)
        # Populate ID to snippet and filename mappings
        for idx, pk in enumerate(mr.primary_keys):
            id_to_snippet[pk] = code_snippets[idx][1]  # Storing the code snippet text
            id_to_filename[pk] = code_snippets[idx][0]  # Storing the filename

        # Create index and load collection after inserting
        create_index(collection)
        load_collection(collection)
    else:
        print("No valid documents to insert.")

def fetch_filename_by_id(doc_id):
    return id_to_filename.get(doc_id)


# Function to query the local ollama instance
def query_ollama(prompt):
    client = OpenAI(
        base_url='http://localhost:11434/v1/',

        # required but ignored
        api_key='ollama',
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                'role': 'system',
                'content': 'You are a code analyser, you recieve multiple code files with their filename at the top and bottom of the code, you will answer questions relating to the code and understand that eaxh snippet belongs to a seperate file',
            },
            {
                'role': 'user',
                'content': prompt,
            }
        ],
        model='llama3',
    )

    return chat_completion


def save_embeddings(embeddings, filename):
    with open(filename, 'wb') as f:
        pickle.dump(embeddings, f)

def load_embeddings(filename):
    try:
        with open(filename, 'rb') as f:
            embeddings = pickle.load(f)
        return embeddings
    except FileNotFoundError:
        return None


# Define the index parameters

# Create the index
def create_index(collection, field_name="embedding"):
    print("Creating index...")
    index_params = {
        "metric_type": "L2",  # Or any other metric suitable for your use case
        "index_type": "IVF_FLAT",  # Change as needed based on your performance requirements
        "params": {"nlist": 100}
    }
    index = Index(collection, field_name, index_params)
    collection.create_index(field_name, index_params)
    print("Index created.")

# Load the collection
def load_collection(collection):
    print("Loading collection...")
    collection.load()
    print("Collection loaded.")


# Global dictionary to hold ID to code snippet mapping
id_to_snippet = {}

def fetch_code_snippet_by_id(result):
    # Return the code snippet corresponding to the given ID
    return f"filename: {result.entity.get('filename')} \n{id_to_snippet.get(result.id)} \nEnd of code snippet"

def generate_query_embedding(query, model="text-embedding-3-large"):
    # Process the query by stripping leading/trailing spaces and replacing internal line breaks
    processed_query = ' '.join(query.strip().split())
    print(f"Processed query: {processed_query}")  # Debugging output
    
    try:
        response = client.embeddings.create(
            input=processed_query,
            model=model
        )
        if response and response.data:
            embedding = response.data[0].embedding
            return embedding
        else:
            print("No data returned from embedding API")
            return None
    except Exception as e:
        print(f"Failed to generate embedding for query: {str(e)}")
        return None


def extract_code_blocks(source_code):
    """
    Parse Python source code, extract functions and classes, and return their code as strings.
    """
    blocks = []

    tree = ast.parse(source_code)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Extract the source code of the block
            start_lineno = node.lineno - 1
            end_lineno = node.end_lineno
            block_code = '\n'.join(source_code.splitlines()[
                                start_lineno:end_lineno])
            block_name = node.name
            blocks.append((block_name, block_code))

    return blocks

def query_gpt4(user_input):
    client = OpenAI(
    api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
    )

    completion = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
        {"role": "system", "content": "You are a code analyser, you recieve multiple code files with their filename at the top and bottom of the code, you will answer questions relating to the code and understand that eaxh snippet belongs to a seperate file"},
        {"role": "user", "content": user_input}
    ]
    )
    return completion.choices[0].message

if "__name__" == "__main__":
    # Example use case: Inserting and querying code embeddings
    embeddings_file = 'uppgift2_embeddings_openAI.pkl'
    code_directory = './codebase/Uppgift_2'  # path to the pandas directory'
    # embeddings = load_embeddings(embeddings_file)
    code_snippets = extract_code_snippets(code_directory)
    collection = create_collection(3072)
    insert_embeddings(code_snippets, embeddings_file, collection)

    # Searching the collection

    # Construct a user query and process it
    user_query = "Explain the file D_hyper_params.py"
    # Simulate a search in Milvus (this is a placeholder)
    search_params = {"metric_type": "L2", "params": {"nprobe": 100}}
    query_embedding = generate_query_embedding(user_query)

    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=10,
        output_fields=["filename"]  # Make sure to include 'filename' to retrieve it
    )

    # Assume each 'hits' contains a list of hit objects which have 'id', 'score', and 'entity'
    sorted_results = []
    for hits in results:
        # Sort hits within each result set in descending order by score
        sorted_hits = sorted(hits, key=lambda hit: hit.score, reverse=True)
        sorted_results.append(sorted_hits)

    # Assuming fetch_code_snippet_by_id function is defined to get the actual code
    relevant_snippets = [fetch_code_snippet_by_id(result) for result in sorted_results[0]]

    context = "\n\n".join(relevant_snippets)
    prompt = f"{context}\n\nUser Question: {user_query}"    

    # Query ollama with constructed prompt
    response = query_gpt4(prompt)
    print(response.content)
