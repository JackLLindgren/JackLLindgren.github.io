import pandas as pd
import re
from collections import defaultdict
import torch
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer, util
import numpy as np  # Import numpy for array handling
import os

# --- Local Sentence-BERT Setup ---
# *** IMPORTANT: Replace this with the actual local path to your downloaded Sentence-BERT model files ***
# This path should point to the directory containing model files like config.json, tokenizer.json, pytorch_model.bin
LOCAL_MODEL_PATH = r"LLM Path Here"

# Path to save/load pre-calculated embeddings (CSV format)
EMBEDDINGS_FILE_PATH = "unique_customer_embeddings.csv"

# Initialize Sentence-BERT model globally
sentence_bert_model = None

# Threshold for Sentence-BERT embedding similarity to consider names as "the same"
EMBEDDING_SIMILARITY_THRESHOLD = 0.85  # Adjust as needed (0.0 - 1.0)


def load_sentence_bert_model():
    """
    Loads the local Sentence-BERT model.
    Attempts to load from LOCAL_MODEL_PATH first. If that fails,
    it tries to load a standard 'all-mpnet-base-v2' model (which may download).
    """
    global sentence_bert_model

    if sentence_bert_model is not None:
        return  # Already loaded

    print(f"Attempting to load Sentence-BERT model from: {LOCAL_MODEL_PATH}...")
    try:
        # Try loading from the user-specified local path first
        sentence_bert_model = SentenceTransformer(LOCAL_MODEL_PATH)
        print("Local Sentence-BERT model loaded successfully from specified path.")
    except Exception as e:
        print(f"Error loading model from {LOCAL_MODEL_PATH}: {e}")
        print("This might be due to an incomplete or incorrect local model directory.")
        print("Attempting to load standard 'all-mpnet-base-v2' model (may download from Hugging Face)...")
        try:
            # Fallback: Load a standard model which will download if not cached
            sentence_bert_model = SentenceTransformer('all-mpnet-base-v2')
            print("Standard 'all-mpnet-base-v2' model loaded successfully.")
        except Exception as e_fallback:
            print(f"Error loading standard Sentence-BERT model: {e_fallback}")
            print(
                "Please ensure sentence-transformers is installed correctly and you have an internet connection for initial download.")
            sentence_bert_model = None  # Indicate complete failure to load

    if sentence_bert_model:
        if torch.cuda.is_available():
            print(f"Sentence-BERT model loaded to GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA not available for Sentence-BERT. Model will run on CPU (slower).")
    else:
        print("Sentence-BERT model could not be loaded. Embeddings functionality will be disabled.")


def get_or_create_embeddings(unique_names_to_encode):
    """
    Loads embeddings from a CSV file if available, otherwise calculates and saves them to CSV.

    Args:
        unique_names_to_encode (list): A list of unique customer names to encode.

    Returns:
        dict: A dictionary mapping customer name to its embedding (NumPy array).
    """
    embeddings_dict = {}
    load_sentence_bert_model()  # Ensure model is loaded

    if sentence_bert_model is None:
        print("Sentence-BERT model not loaded. Cannot generate embeddings.")
        return {}

    if os.path.exists(EMBEDDINGS_FILE_PATH):
        print(f"Attempting to load embeddings from {EMBEDDINGS_FILE_PATH}...")
        try:
            loaded_df = pd.read_csv(EMBEDDINGS_FILE_PATH)
            # Convert embedding strings back to NumPy arrays
            loaded_df['Embedding_Vector'] = loaded_df['Embedding_Vector'].apply(
                lambda x: np.array(list(map(float, x.strip('[]').split(','))))
            )
            loaded_embeddings_dict = pd.Series(
                loaded_df.Embedding_Vector.values, index=loaded_df.CustomerName
            ).to_dict()
            loaded_unique_names_list = loaded_df['CustomerName'].tolist()

            # Check if loaded embeddings cover all current unique names
            if set(loaded_unique_names_list) == set(unique_names_to_encode):
                embeddings_dict = loaded_embeddings_dict
                print("Embeddings loaded successfully from file.")
                return embeddings_dict
            else:
                print("Loaded embeddings are for a different set of names or incomplete. Recalculating...")
        except Exception as e:
            print(f"Error loading embeddings from file: {e}. Recalculating embeddings...")
            embeddings_dict = {}  # Reset if loading fails

    print(f"Calculating embeddings for {len(unique_names_to_encode)} unique names. This may take a while...")
    batch_size = 128

    all_embeddings = []
    for i in range(0, len(unique_names_to_encode), batch_size):
        batch_names = unique_names_to_encode[i:i + batch_size]
        batch_embeddings = sentence_bert_model.encode(batch_names, convert_to_tensor=False, show_progress_bar=False)
        all_embeddings.extend(batch_embeddings)
        print(f"  Processed {i + len(batch_names)}/{len(unique_names_to_encode)} embeddings.")

    # Map original unique names to their embeddings
    embeddings_dict = {name: emb for name, emb in zip(unique_names_to_encode, all_embeddings)}

    print(f"Embeddings calculation complete. Saving to {EMBEDDINGS_FILE_PATH}...")
    try:
        # Create a DataFrame for saving
        df_to_save = pd.DataFrame({
            'CustomerName': list(embeddings_dict.keys()),
            'Embedding_Vector': [emb.tolist() for emb in embeddings_dict.values()]  # Convert NumPy array to list for CSV
        })
        df_to_save.to_csv(EMBEDDINGS_FILE_PATH, index=False)
        print("Embeddings saved successfully to CSV.")
    except Exception as e:
        print(f"Error saving embeddings to file: {e}")

    return embeddings_dict


def process_customer_data(df, use_embedding_similarity=False, fuzzywuzzy_threshold=85,
                          embedding_similarity_threshold=0.85):
    """
    Processes customer data to assign unique IDs and deduplicate using similarity,
    without any name normalization. It uses the raw 'CustomerName' directly.
    """
    print(f"Processing {len(df)} rows of customer data using raw customer names.")

    df['CustomerName_Raw'] = df['CustomerName'].astype(str)

    unique_raw_names = df['CustomerName_Raw'].unique().tolist()

    name_to_base_id = {name: i + 1 for i, name in enumerate(unique_raw_names)}
    df['Unique Identifier'] = df['CustomerName_Raw'].map(name_to_base_id).astype(int)

    df['Embedding_Similarity_Score'] = pd.NA

    print(f"Found {len(unique_raw_names)} unique raw names for initial identification.")

    embedding_group_min_similarity = {id_val: 1.0 for id_val in df['Unique Identifier'].unique()}

    embeddings_map = {}
    if use_embedding_similarity:
        embeddings_map = get_or_create_embeddings(unique_raw_names)
        if not embeddings_map:
            print("Failed to get embeddings. Reverting to FuzzyWuzzy-only matching.")
            use_embedding_similarity = False

    if not use_embedding_similarity:
        print("Using FuzzyWuzzy-only string matching (token_set_ratio as direct merge criteria)...")
    else:
        print(f"Using Embedding Similarity for matching (threshold={embedding_similarity_threshold})...")

    id_mapping = {id_val: id_val for id_val in df['Unique Identifier'].unique()}

    unique_records_for_comparison = df[['CustomerName', 'Unique Identifier', 'CustomerName_Raw']].drop_duplicates().values.tolist()
    print(
        f"Considering {len(unique_records_for_comparison)} distinct (original name, initial ID, raw name) pairs for comparison.")

    merge_count = 0
    comparison_method = "Embedding Similarity" if use_embedding_similarity else "FuzzyWuzzy"

    for i in range(len(unique_records_for_comparison)):
        name1_orig, id1_initial, name1_raw = unique_records_for_comparison[i]
        current_id1 = id_mapping[id1_initial]

        for j in range(i + 1, len(unique_records_for_comparison)):
            name2_orig, id2_initial, name2_raw = unique_records_for_comparison[j]
            current_id2 = id_mapping[id2_initial]

            if current_id1 != current_id2:
                if use_embedding_similarity:
                    emb1 = embeddings_map.get(name1_raw)
                    emb2 = embeddings_map.get(name2_raw)

                    if emb1 is None or emb2 is None:
                        print(f"Warning: Embedding not found for '{name1_raw}' or '{name2_raw}'. Skipping comparison.")
                        continue

                    embedding_score = util.cos_sim(torch.tensor(emb1), torch.tensor(emb2)).item()

                    # FIX: Changed 'similarity_threshold' to 'embedding_similarity_threshold'
                    if embedding_score >= embedding_similarity_threshold:
                        print(
                            f"Embedding confirmed '{name1_orig}' and '{name2_orig}' are the same (Similarity: {embedding_score:.2f}). Merging ID {current_id2} into {current_id1}.")
                        merge_count += 1
                        merge_to_id = min(current_id1, current_id2)
                        merge_from_id = max(current_id1, current_id2)

                        for k, v in id_mapping.items():
                            if v == merge_from_id:
                                id_mapping[k] = merge_to_id
                        for item_idx in range(j, len(unique_records_for_comparison)):
                            # This needs to update the list in place with the correct merged ID
                            # unique_records_for_comparison[item_idx][1] is the initial ID, which needs to be re-mapped
                            if id_mapping[unique_records_for_comparison[item_idx][1]] == merge_from_id:
                                unique_records_for_comparison[item_idx][1] = merge_to_id


                        embedding_group_min_similarity[merge_to_id] = min(
                            embedding_group_min_similarity.get(merge_to_id, 1.0), embedding_score
                        )
                        embedding_group_min_similarity[merge_from_id] = embedding_group_min_similarity[merge_to_id]

                else:
                    fuzzy_score = fuzz.token_set_ratio(name1_raw.lower(), name2_raw.lower())
                    if fuzzy_score >= fuzzywuzzy_threshold:
                        print(
                            f"FuzzyWuzzy match found ({fuzzy_score}%): '{name1_orig}' and '{name2_orig}'. Merging ID {current_id2} into {current_id1}.")
                        merge_count += 1
                        for k, v in id_mapping.items():
                            if v == current_id2:
                                id_mapping[k] = current_id1
                        for item_idx in range(j, len(unique_records_for_comparison)):
                            if unique_records_for_comparison[item_idx][1] == current_id2:
                                unique_records_for_comparison[item_idx][1] = current_id1

    print(f"Total merges performed: {merge_count}")

    df['Unique Identifier'] = df['Unique Identifier'].map(id_mapping).astype(int)

    final_id_renumbering = {}
    unique_final_ids_sorted = sorted(df['Unique Identifier'].unique())
    for new_idx, old_id in enumerate(unique_final_ids_sorted):
        final_id_renumbering[old_id] = new_idx + 1

    df['Unique Identifier'] = df['Unique Identifier'].map(final_id_renumbering)

    reverse_renumbering = {v: k for k, v in final_id_renumbering.items()}

    for index, row in df.iterrows():
        final_uid = row['Unique Identifier']
        original_uid_before_renumbering = reverse_renumbering.get(final_uid)

        if use_embedding_similarity:
            if original_uid_before_renumbering in embedding_group_min_similarity:
                df.at[index, 'Embedding_Similarity_Score'] = round(
                    embedding_group_min_similarity[original_uid_before_renumbering] * 100, 2)
            else:
                df.at[index, 'Embedding_Similarity_Score'] = 100.0
        else:
            df.at[index, 'Embedding_Similarity_Score'] = pd.NA

    print(f"Final number of unique customer identifiers after merging: {len(unique_final_ids_sorted)}")
    df = df.drop(columns=['CustomerName_Raw'])

    return df

# --- Main execution ---
if __name__ == "__main__":
    input_excel_file = r"C:\Users\JackLindgren\OneDrive - Burwell Material Handling\Desktop\Data\Company Naming Problem set.xlsx"
    output_excel_file = "customers_unique_ids_raw_names_csv_embeddings.xlsx"  # New output file name

    try:
        df = pd.read_excel(input_excel_file)

        processed_df = process_customer_data(df.copy(),
                                             use_embedding_similarity=True,
                                             fuzzywuzzy_threshold=85,
                                             embedding_similarity_threshold=EMBEDDING_SIMILARITY_THRESHOLD)

        processed_df.to_excel(output_excel_file, index=False)
        print(f"Successfully created '{output_excel_file}' with unique customer IDs and Embedding Similarity Score.")
        print("\nProcessed Data Preview (first 10 rows):")
        print(processed_df.head(10))

    except FileNotFoundError:
        print(f"Error: The file '{input_excel_file}' was not found.")
        print("Please check the path and ensure the file exists.")
    except Exception as e:
        print(f"An error occurred: {e}")