# query_service.py
import numpy as np
from annoy import AnnoyIndex
import os
from sentence_transformers import SentenceTransformer, util
import pandas as pd

# --- Configuration (must match what was used to build the index) ---
# IMPORTANT: Replace this with the actual local path to your downloaded Sentence-BERT model files
# This path should point to the directory containing model files like config.json, tokenizer.json, pytorch_model.bin
LOCAL_MODEL_PATH = r"Insert Model Path"

ANNOY_INDEX_FILE_PATH = "customer_annoy_index.ann"
EMBEDDING_DIM = 768 # This should match the dimension of your Sentence-BERT embeddings
ANNOY_METRIC = 'angular' # Use 'angular' for cosine similarity

# Path to the CSV that contains the original names corresponding to Annoy's internal IDs
# This is crucial to map Annoy's integer IDs back to actual customer names.
EMBEDDINGS_CSV_PATH = "unique_customer_embeddings.csv"

# Global variables to store loaded model, index, and names list
global_sentence_bert_model = None
global_annoy_index = None
global_original_names_list = None

# --- Load Sentence-BERT Model (only once) ---
def load_sentence_bert_model_global():
    """Loads the local Sentence-BERT model globally."""
    global global_sentence_bert_model
    if global_sentence_bert_model is None:
        try:
            global_sentence_bert_model = SentenceTransformer(LOCAL_MODEL_PATH)
            print("Sentence-BERT model loaded successfully.")
        except Exception as e:
            print(f"Error loading Sentence-BERT model: {e}")
            print("Cannot encode new query names without the model.")
            global_sentence_bert_model = None
    return global_sentence_bert_model

# --- Load Original Names (only once) ---
def load_original_names_map_global(csv_path):
    """
    Loads customer names from the embeddings CSV globally.
    """
    global global_original_names_list
    if global_original_names_list is None:
        if not os.path.exists(csv_path):
            print(f"Error: Original embeddings CSV not found at {csv_path}. Cannot map Annoy IDs to names.")
            return None
        try:
            df_embeddings = pd.read_csv(csv_path)
            global_original_names_list = df_embeddings['CustomerName'].tolist()
            print("Original names map loaded successfully.")
        except Exception as e:
            print(f"Error loading original names from CSV: {e}")
            global_original_names_list = None
    return global_original_names_list

# --- Load Annoy Index (only once) ---
def load_annoy_index_global():
    """Loads the Annoy index globally."""
    global global_annoy_index
    if global_annoy_index is None:
        global_annoy_index = AnnoyIndex(EMBEDDING_DIM, ANNOY_METRIC)
        if not os.path.exists(ANNOY_INDEX_FILE_PATH):
            print(f"Error: Annoy index file not found at {ANNOY_INDEX_FILE_PATH}. Please run the clustering script first to build and save it.")
            global_annoy_index = None
            return None
        try:
            global_annoy_index.load(ANNOY_INDEX_FILE_PATH)
            print(f"Annoy index loaded from {ANNOY_INDEX_FILE_PATH}.")
        except Exception as e:
            print(f"Error loading Annoy index: {e}")
            global_annoy_index = None
    return global_annoy_index


# --- Main Query Function ---
def query_customer_names(query_name, n_results=5, min_cos_similarity=0.7):
    """
    Queries the Annoy index for nearest neighbors of a given query name.
    Prioritizes and reports exact textual matches first with a similarity of 1.0.
    Assumes global_sentence_bert_model, global_annoy_index, and global_original_names_list are loaded.

    Args:
        query_name (str): The customer name to query for.
        n_results (int): Number of nearest neighbors to retrieve (excluding exact match if found).
        min_cos_similarity (float): Minimum cosine similarity to consider a neighbor relevant.
    """
    # Ensure global assets are available
    if global_sentence_bert_model is None or global_annoy_index is None or global_original_names_list is None:
        return [] # Return empty list if not loaded

    print(f"\n--- Querying for: '{query_name}' ---")

    # --- Step 1: Check for Exact Letter-by-Letter Match ---
    found_exact_match = False
    exact_match_name = None
    for name_in_db in global_original_names_list:
        if name_in_db == query_name: # Case-sensitive exact match
            found_exact_match = True
            exact_match_name = name_in_db
            break

    if found_exact_match:
        print(f"**Exact match found in dataset:** '{exact_match_name}' (Cosine Similarity: 1.0000) ✅")
        results = [{'name': exact_match_name, 'similarity': 1.0}]
        n_results_for_annoy = n_results - 1 # Adjust for already found exact match
    else:
        results = []
        n_results_for_annoy = n_results


    # --- Step 2: Proceed with Semantic Search using Annoy for Similar Names ---
    # Only proceed if we still need more results or no exact match was found
    if n_results_for_annoy > 0:
        # Encode the query name into an embedding
        # print(f"Encoding query name: '{query_name}' for similarity search...") # Suppress for frontend
        try:
            query_embedding = global_sentence_bert_model.encode(query_name, convert_to_tensor=False)
            # print("Query name encoded.") # Suppress for frontend
        except Exception as e:
            print(f"Error encoding query name: {e}")
            return results # Return current results if encoding fails

        # Find nearest neighbors in the Annoy index
        # print(f"Finding {n_results_for_annoy} nearest semantic neighbors...") # Suppress for frontend
        # Add a buffer to n_results_for_annoy to ensure we get enough after filtering
        neighbor_ids, distances = global_annoy_index.get_nns_by_vector(
            query_embedding, n=n_results_for_annoy + 5, include_distances=True # Fetch a few more than needed
        )

        # Process Annoy results
        for neighbor_id, dist in zip(neighbor_ids, distances):
            cos_similarity = 1 - (dist**2 / 2)
            neighbor_name = global_original_names_list[neighbor_id]

            # Filter:
            # 1. Ensure it meets the minimum similarity threshold
            # 2. Exclude the exact match if it was already found and added
            # 3. Prevent duplicate entries in results list
            if (cos_similarity >= min_cos_similarity and
                neighbor_name != exact_match_name and
                {'name': neighbor_name, 'similarity': cos_similarity} not in results):
                results.append({
                    'name': neighbor_name,
                    'similarity': cos_similarity
                })

            if len(results) >= n_results + (1 if found_exact_match else 0): # Stop once we have enough total results
                break

    # Sort results by similarity score, descending
    results_sorted = sorted(results, key=lambda x: x['similarity'], reverse=True)
    # Ensure we only display up to the requested n_results + (1 for exact if present)
    display_limit = n_results + (1 if found_exact_match else 0)

    # Return the results to the Flask app
    return results_sorted[:display_limit]