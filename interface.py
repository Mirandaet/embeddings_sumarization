import vector_db_embeddings as vbd
import os
from summarization import CodeSummarizer

def query_vdb(user_query, code_directory="../codebase/Uppgift_2", create_new_embeddings=False):
    # embeddings = load_embeddings(embeddings_file)
    
    embeddings_file = code_directory + "_embeddings.pkl"
    
    if create_new_embeddings == "true":
        try:
            os.remove(embeddings_file)
            print(f"File '{embeddings_file}' has been removed successfully.")
        except FileNotFoundError:
            print(f"File '{embeddings_file}' not found.")
        except PermissionError:
            print(f"Permission denied: Unable to delete '{embeddings_file}'.")
        except Exception as e:
            print(f"Error occurred while trying to delete '{embeddings_file}': {e}")

    code_snippets = vbd.extract_code_snippets(code_directory)
    collection = vbd.create_collection(3072)
    vbd.insert_embeddings(code_snippets, embeddings_file, collection)

    # Searching the collection

    # Construct a user query and process it
    # Simulate a search in Milvus (this is a placeholder)
    search_params = {"metric_type": "L2", "params": {"nprobe": 100}}
    query_embedding = vbd.generate_query_embedding(user_query)
    

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
    relevant_snippets = [vbd.fetch_code_snippet_by_id(result) for result in sorted_results[0]]

    context = "\n\n".join(relevant_snippets)
    prompt = f"{context}\n\nUser Question: {user_query}"    

    # Query ollama with constructed prompt
    response = vbd.query_gpt4(prompt)
    return response.content


def get_summary(path):
    api_key = os.environ['OPENAI_API_KEY']
    model = 'gpt-4-instruct'  # Or any other suitable model
    summarizer = CodeSummarizer(model, api_key)

    if os.path.isdir(path):
        # If the path is a directory, process all Python files in the directory
        summaries = summarizer.process_directory(path)
        return summaries
    elif os.path.isfile(path) and path.endswith('.py'):
        # If the path is a file, summarize only that file
        summary = summarizer.query_gpt4(path)
        return summary
    else:
        raise ValueError("Provided path must be a directory or a Python file.")



