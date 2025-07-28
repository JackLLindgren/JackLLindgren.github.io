import pandas as pd
import numpy as np
from annoy import AnnoyIndex
import os
import ast # For safely evaluating string representation of lists

# --- Configuration ---
EMBEDDINGS_FILE_PATH = "Insert Embeddings Path"
ANNOY_INDEX_FILE_PATH = "customer_annoy_index.ann" # New: Path to save/load Annoy index
OUTPUT_CLUSTERS_FILE = "Insert Output Name"

# Dimensionality of your embeddings (768 for all-mpnet-base-v2)
EMBEDDING_DIM = 768

# Distance metric for Annoy (angular is equivalent to cosine similarity on normalized vectors)
# Annoy's angular distance is sqrt(2 * (1 - cos_similarity))
# So, cos_similarity = 1 - (angular_distance**2 / 2)
ANNOY_METRIC = 'angular'

# Number of trees to build for the Annoy index (higher = more accurate, slower build)
N_TREES = 100

# Number of nearest neighbors to fetch for each item
N_NEIGHBORS_TO_FETCH = 10

# Similarity threshold to consider two items part of the same cluster (cosine similarity)
# This should be consistent with your deduplication threshold.
CLUSTERING_COS_SIM_THRESHOLD = 0.9

# --- Function to load embeddings ---
def load_embeddings_from_csv(file_path):
    """
    Loads customer names and their embeddings from a CSV file.
    Converts embedding strings back to NumPy arrays.
    """
    if not os.path.exists(file_path):
        print(f"Error: Embeddings file not found at {file_path}")
        return None, None

    print(f"Loading embeddings from {file_path}...")
    try:
        df_embeddings = pd.read_csv(file_path)
        # Convert the string representation of list back to actual list/numpy array
        # Using ast.literal_eval is safer than eval() for untrusted input
        df_embeddings['Embedding_Vector'] = df_embeddings['Embedding_Vector'].apply(
            lambda x: np.array(ast.literal_eval(x))
        )
        names = df_embeddings['CustomerName'].tolist()
        embeddings = df_embeddings['Embedding_Vector'].tolist()
        print(f"Successfully loaded {len(names)} embeddings.")
        return names, embeddings
    except Exception as e:
        print(f"Error loading embeddings from CSV: {e}")
        return None, None

# --- Main Clustering Logic ---
def create_clusters_with_annoy(names, embeddings, embedding_dim, metric, n_trees,
                                n_neighbors_to_fetch, clustering_cos_sim_threshold,
                                save_index=True, load_index=True):
    """
    Creates an Annoy index and performs a simple connectivity-based clustering
    to group similar customer names. It can also save and load the Annoy index.

    Args:
        names (list): List of customer names.
        embeddings (list): List of embedding vectors (NumPy arrays).
        embedding_dim (int): Dimensionality of embedding vectors.
        metric (str): Annoy distance metric.
        n_trees (int): Number of trees for Annoy index.
        n_neighbors_to_fetch (int): Number of nearest neighbors to fetch.
        clustering_cos_sim_threshold (float): Cosine similarity threshold for clustering.
        save_index (bool): Whether to save the Annoy index after building.
        load_index (bool): Whether to attempt loading the Annoy index if it exists.

    Returns:
        list: A list of lists, where each inner list represents a cluster of customer names.
    """
    annoy_index = AnnoyIndex(embedding_dim, metric)
    index_loaded = False

    if load_index and os.path.exists(ANNOY_INDEX_FILE_PATH):
        print(f"Attempting to load Annoy index from {ANNOY_INDEX_FILE_PATH}...")
        try:
            annoy_index.load(ANNOY_INDEX_FILE_PATH)
            print("Annoy index loaded successfully.")
            index_loaded = True
        except Exception as e:
            print(f"Error loading Annoy index: {e}. Rebuilding index...")
            index_loaded = False

    if not index_loaded:
        if not names or not embeddings:
            print("No names or embeddings provided for building index.")
            return []

        print(f"Building Annoy index with {len(embeddings)} items and {embedding_dim} dimensions...")
        for i, emb in enumerate(embeddings):
            annoy_index.add_item(i, emb)
        annoy_index.build(n_trees)
        print(f"Annoy index built with {n_trees} trees.")

        if save_index:
            try:
                annoy_index.save(ANNOY_INDEX_FILE_PATH)
                print(f"Annoy index saved to {ANNOY_INDEX_FILE_PATH}.")
            except Exception as e:
                print(f"Error saving Annoy index: {e}")

    clusters = []
    visited = set() # Keep track of items already assigned to a cluster

    print("Finding nearest neighbors and forming clusters...")
    # It's important to iterate over the original indices (0 to len(names)-1)
    # as annoy_index.get_nns_by_item works with these internal IDs.
    for i in range(len(names)):
        if i in visited:
            continue

        current_cluster_members = []
        queue = [i] # Start BFS from this unvisited item
        visited.add(i)

        while queue:
            current_idx = queue.pop(0) # Use pop(0) for BFS
            current_cluster_members.append(names[current_idx])

            # Get nearest neighbors for the current item
            # Annoy returns IDs and distances (angular distance)
            # Fetch N_NEIGHBORS_TO_FETCH + 1 to account for the item itself
            neighbor_ids, distances = annoy_index.get_nns_by_item(
                current_idx, n=n_neighbors_to_fetch + 1, include_distances=True
            )

            # Remove self from neighbors if present (it's typically the first result)
            if current_idx in neighbor_ids:
                self_idx = neighbor_ids.index(current_idx)
                neighbor_ids.pop(self_idx)
                distances.pop(self_idx)

            for neighbor_id, dist in zip(neighbor_ids, distances):
                # Convert Annoy's angular distance to cosine similarity
                # angular_distance = sqrt(2 * (1 - cos_similarity))
                # cos_similarity = 1 - (angular_distance**2 / 2)
                cos_similarity = 1 - (dist**2 / 2)

                if neighbor_id not in visited and cos_similarity >= clustering_cos_sim_threshold:
                    visited.add(neighbor_id)
                    queue.append(neighbor_id)

        if current_cluster_members:
            clusters.append(current_cluster_members)

    print(f"Found {len(clusters)} clusters.")
    return clusters

# --- Main execution ---
if __name__ == "__main__":
    # Load embeddings from the CSV file
    customer_names, customer_embeddings = load_embeddings_from_csv(EMBEDDINGS_FILE_PATH)

    if customer_names and customer_embeddings:
        # Create clusters, attempting to load index first, then saving if rebuilt
        found_clusters = create_clusters_with_annoy(
            customer_names,
            customer_embeddings,
            EMBEDDING_DIM,
            ANNOY_METRIC,
            N_TREES,
            N_NEIGHBORS_TO_FETCH,
            CLUSTERING_COS_SIM_THRESHOLD,
            save_index=True,  # Set to True to save the index after building
            load_index=True   # Set to True to attempt loading the index
        )

        # Save clusters to a text file
        try:
            with open(OUTPUT_CLUSTERS_FILE, 'w', encoding='utf-8') as f:
                f.write(f"--- Customer Clusters (Annoy-based Connectivity) ---\n\n")
                f.write(f"Parameters:\n")
                f.write(f"  Embeddings File: {EMBEDDINGS_FILE_PATH}\n")
                f.write(f"  Annoy Index File: {ANNOY_INDEX_FILE_PATH}\n") # Added to parameters
                f.write(f"  Embedding Dimension: {EMBEDDING_DIM}\n")
                f.write(f"  Annoy Metric: {ANNOY_METRIC}\n")
                f.write(f"  Annoy Trees: {N_TREES}\n")
                f.write(f"  Neighbors Fetched Per Item: {N_NEIGHBORS_TO_FETCH}\n")
                f.write(f"  Clustering Cosine Similarity Threshold: {CLUSTERING_COS_SIM_THRESHOLD:.2f}\n\n")
                f.write(f"Total Clusters Found: {len(found_clusters)}\n\n")

                # Sort clusters by size (largest first) for better readability
                found_clusters_sorted = sorted(found_clusters, key=len, reverse=True)

                for i, cluster in enumerate(found_clusters_sorted):
                    if len(cluster) > 1: # Only show clusters with more than one member
                        f.write(f"--- Cluster {i+1} (Size: {len(cluster)}) ---\n")
                        for name in cluster:
                            f.write(f"- {name}\n")
                        f.write("\n")
            print(f"Clusters saved to '{OUTPUT_CLUSTERS_FILE}'.")
        except Exception as e:
            print(f"Error saving clusters to file: {e}")
    else:
        print("Could not load embeddings. Clustering aborted.")
