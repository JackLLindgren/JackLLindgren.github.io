import pandas as pd
import numpy as np
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer, util
import os
from collections import defaultdict
import re
import time
import random
import json
import socket
import asyncio
import aiohttp

# --- Configuration ---
INPUT_EXCEL_FILE = 'Insert Excel file Path'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_EXCEL_FILE = os.path.join(SCRIPT_DIR, "Insert Output Name")

COMPANY_NAME_COLUMN = 'CustomerName'
CUSTOMER_CODE_COLUMN = 'CustomerCode'
CUSTOMER_PARENT_COLUMN = 'CustomerParent'
CUSTOMER_ROLLUP_COLUMN = 'CustomerRollup'
FLAGGED_COLUMN = 'Flagged'

# Sentence-BERT and Annoy model configuration
LOCAL_MODEL_PATH = r"C:\LLm\mpnet"
ANNOY_INDEX_FILE_PATH = os.path.join(SCRIPT_DIR, "customer_annoy_index.ann")
EMBEDDINGS_CSV_PATH = os.path.join(SCRIPT_DIR, "unique_customer_embeddings.csv")
EMBEDDING_DIM = 768
ANNOY_METRIC = 'angular'

ANNOY_CANDIDATE_SIMILARITY_THRESHOLD = 0.5

# --- Gemini API Configuration ---
GEMINI_MODEL_NAME = "gemini-1.5-flash"
API_CALL_DELAY_SECONDS = 0.01  # Current aggressive setting
MAX_RETRIES = 10
INITIAL_RETRY_DELAY = 2

CONCURRENT_API_CALLS_LIMIT = 40  # Current aggressive setting

# --- Checkpointing Configuration ---
CHECKPOINT_DIR_ROLLUP = os.path.join(SCRIPT_DIR, "rollup_checkpoints")
CACHE_CHECKPOINT_FILE_ROLLUP = os.path.join(CHECKPOINT_DIR_ROLLUP, "llm_cache_rollup.json")
PROGRESS_CHECKPOINT_FILE_ROLLUP = os.path.join(CHECKPOINT_DIR_ROLLUP, "progress_rollup.json")
CHECKPOINT_INTERVAL_COMPANIES = 100  # Checkpoint after this many companies are processed

# --- Global Models and Data ---
sentence_bert_model = None
annoy_index = None
unique_customer_names = []
unique_name_to_embedding = {}
llm_decision_cache = {}
llm_decision_count = 0


# --- Helper Functions ---

def check_internet_connection(timeout=5):
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=timeout)
        return True
    except OSError:
        return False


def save_rollup_checkpoint(processed_codes_count):
    os.makedirs(CHECKPOINT_DIR_ROLLUP, exist_ok=True)
    try:
        # Convert frozenset keys to sorted lists for consistent JSON serialization
        serializable_cache = {
            json.dumps(sorted(list(k))): v
            for k, v in llm_decision_cache.items()
        }
        with open(CACHE_CHECKPOINT_FILE_ROLLUP, 'w') as f:
            json.dump(serializable_cache, f, indent=4)  # Added indent for readability

        with open(PROGRESS_CHECKPOINT_FILE_ROLLUP, 'w') as f:
            json.dump({"processed_codes_count": processed_codes_count}, f)

        print(f"\n--- Rollup Checkpoint saved at {processed_codes_count} companies processed. ---", flush=True)
    except Exception as e:
        print(f"Error saving rollup checkpoint: {e}. Check directory permissions or disk space.", flush=True)


def load_rollup_checkpoint():
    global llm_decision_cache, llm_decision_count
    processed_codes_count = 0

    if os.path.exists(CACHE_CHECKPOINT_FILE_ROLLUP) and \
            os.path.exists(PROGRESS_CHECKPOINT_FILE_ROLLUP):
        try:
            with open(CACHE_CHECKPOINT_FILE_ROLLUP, 'r') as f:
                deserialized_cache = json.load(f)
                llm_decision_cache = {
                    frozenset(json.loads(k)): tuple(v) if isinstance(v, list) else v
                    # Ensure tuple for value consistency
                    for k, v in deserialized_cache.items()
                }

            with open(PROGRESS_CHECKPOINT_FILE_ROLLUP, 'r') as f:
                progress_data = json.load(f)
                processed_codes_count = progress_data.get("processed_codes_count", 0)

            print("\n--- Rollup Checkpoint loaded successfully. Resuming from last state. ---", flush=True)
            llm_decision_count = len(llm_decision_cache)  # Reset LLM call count from loaded cache
            return processed_codes_count
        except Exception as e:
            print(f"Error loading rollup checkpoint, starting fresh: {e}", flush=True)
            llm_decision_cache.clear()
            llm_decision_count = 0
            return 0
    return 0


async def ask_gemini_if_same_company_async(session, name1: str, name2: str) -> tuple[bool, str]:
    global llm_decision_count
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return False, "Gemini API key not set."

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent?key={api_key}"

    prompt_text = f"""Are these two company names referring to the exact same company or a sub-entity of the same larger organization?
    For companies that are different locations or divisions of the same main company (e.g., store numbers, factory branches), consider them the same company.
    For military-based names, consider them the same company if they belong to the same branch (e.g., all US Army entities are one company), regardless of specific unit, installation, or division.

    Respond with ONLY 'Decision: [YES/NO]\\nExplanation: [Brief, 1-2 sentence reason for your decision]'.

    Examples:
    Name 1: "Microsoft Corp."
    Name 2: "Microsoft Corporation"
    Decision: YES
    Explanation: "Corp." is a common abbreviation for "Corporation", referring to the same entity.

    Name 1: "Alphabet Inc."
    Name 2: "Google LLC"
    Decision: NO
    Explanation: Alphabet Inc. is the parent company of Google LLC; they are distinct legal entities.

    Name 1: "Walmart Store #1234"
    Name 2: "Walmart Supercenter 5678"
    Decision: YES
    Explanation: These are different store numbers/types under the same larger Walmart retail entity.

    Name 1: "Apple Store, Lincoln Park"
    Name 2: "Apple Retail, Chicago"
    Decision: YES
    Explanation: These are different retail locations of the same Apple Inc. company.

    Name 1: "US Army Fort Bragg"
    Name 2: "Department of the Army Recruitment Center"
    Decision: YES
    Explanation: Both entities belong to the same military branch, the US Army.

    Name 1: "US Air Force Base Davis-Monthan"
    Name 2: "USAF Logistics Squadron"
    Decision: YES
    Explanation: Both entities are part of the US Air Force branch.

    Name 1: "US Navy Shipyard Norfolk"
    Name 2: "United States Marine Corps Training Base"
    Decision: NO
    Explanation: The US Navy and Marine Corps are distinct branches, though under the same department.

    Name 1: "GE Appliances"
    Name 2: "General Electric Power"
    Decision: NO
    Explanation: While both are part of General Electric's history, they often operate as distinct business units or have been divested, making them separate for many contexts.

    Name 1: "GE"
    Name 2: "General Electric"
    Decision: YES
    Explanation: "GE" is a widely recognized acronym for "General Electric."

    Name 1: "ABC Company"
    Name 2: "ABC Co."
    Decision: YES
    Explanation: "Co." is a common abbreviation for "Company", referring to the same entity.

    Name 1: "{name1}"
    Name 2: "{name2}"
    Decision:
    """

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt_text}]
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 100
        }
    }

    retries = 0
    while retries < MAX_RETRIES:
        if not check_internet_connection():
            print(f"  Internet connection lost. Waiting for connection to restore...", flush=True)
            while not check_internet_connection():
                await asyncio.sleep(30)
            retries = 0
            continue

        try:
            await asyncio.sleep(API_CALL_DELAY_SECONDS + random.uniform(0, API_CALL_DELAY_SECONDS / 2))

            async with session.post(api_url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
                response.raise_for_status()
                response_json = await response.json()

            response_text = response_json.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get(
                'text', '').strip()

            match = re.search(r"Decision:\s*(YES|NO)\s*(?:Explanation:\s*(.*))?", response_text,
                              re.IGNORECASE | re.DOTALL)

            llm_decision = False
            llm_explanation = "Gemini output parsing failed or unexpected format."

            if match:
                decision_str = match.group(1).strip().upper()
                llm_decision = (decision_str == 'YES')
                if match.group(2) is not None:
                    llm_explanation = match.group(2).strip()
                else:
                    after_decision = re.sub(r"Decision:\s*(YES|NO)", "", response_text, flags=re.IGNORECASE).strip()
                    if after_decision:
                        llm_explanation = f"Inferred explanation: {after_decision}"
                    else:
                        llm_explanation = "No specific explanation provided by Gemini."
            else:
                cleaned_response_text = response_text.replace('\n', ' ').replace('\r', '')

                if "yes" in cleaned_response_text.lower() and "no" not in cleaned_response_text.lower() and len(
                        cleaned_response_text) < 50:
                    llm_decision = True
                    llm_explanation = f"Inferred 'YES' from Gemini's raw response: {cleaned_response_text[:50]}..."
                elif "no" in cleaned_response_text.lower() and "yes" not in cleaned_response_text.lower() and len(
                        cleaned_response_text) < 50:
                    llm_decision = False
                    llm_explanation = f"Inferred 'NO' from Gemini's raw response: {cleaned_response_text[:50]}..."
                else:
                    llm_decision = False
                    llm_explanation = f"Gemini parsing failed. Raw response: {cleaned_response_text}"

                print(
                    f"  Warning: Gemini output format unexpected for '{name1}' vs '{name2}'. Raw: '{cleaned_response_text[:200]}...' Using fallback parsing.",
                    flush=True)

            llm_decision_count += 1
            return llm_decision, llm_explanation

        except aiohttp.ClientResponseError as e:
            retries += 1
            print(
                f"  Gemini API HTTP Error for '{name1}' vs '{name2}' (Retry {retries}/{MAX_RETRIES}): Status {e.status}, Message: {e.message}",
                flush=True)
            if e.status == 429:  # Too Many Requests
                retry_delay = INITIAL_RETRY_DELAY * (2 ** retries) + random.uniform(0, 1)
                print(f"  Rate limit hit. Waiting {retry_delay:.2f} seconds before retrying...", flush=True)
                await asyncio.sleep(retry_delay)
            elif e.status == 503:  # Service Unavailable (overload)
                retry_delay = INITIAL_RETRY_DELAY * (2 ** retries) + random.uniform(0, 1)
                print(f"  Service overloaded (503). Waiting {retry_delay:.2f} seconds before retrying...", flush=True)
                await asyncio.sleep(retry_delay)
            else:  # Other HTTP errors
                await asyncio.sleep(INITIAL_RETRY_DELAY)
        except aiohttp.ClientError as e:
            retries += 1
            print(f"  Gemini API Network Error for '{name1}' vs '{name2}' (Retry {retries}/{MAX_RETRIES}): {e}",
                  flush=True)
            await asyncio.sleep(INITIAL_RETRY_DELAY)
        except Exception as e:
            retries += 1
            print(f"  Unexpected error for '{name1}' vs '{name2}' (Retry {retries}/{MAX_RETRIES}): {e}", flush=True)
            await asyncio.sleep(INITIAL_RETRY_DELAY)

    return False, f"Failed after {MAX_RETRIES} retries due to Gemini API errors."


def analyze_for_flagging(name1: str, name2: str, llm_explanation: str, is_llm_same: bool) -> str:
    flags = set()

    has_complex_numbers = lambda name: bool(re.search(r'\d{4,}', name)) and (
            sum(c.isdigit() for c in name) / len(name) > 0.6) and len(name) > 5
    unusual_char_pattern = re.compile(r'[^\w\s.,&\'"-/]')
    has_unusual_chars = lambda name: bool(unusual_char_pattern.search(name)) and (
            len(re.findall(unusual_char_pattern, name)) >= 1)

    if has_complex_numbers(name1) or has_complex_numbers(name2):
        flags.add("Complex Numeric ID/Name")
    if has_unusual_chars(name1) or has_unusual_chars(name2):
        flags.add("Unusual Characters/Format")

    is_short_and_ambiguous = lambda name: (len(name.replace(" ", "")) <= 4 and
                                           not re.fullmatch(r"^[A-Z&]{2,4}$", name.replace(" ", "")) and
                                           not name.lower() in ['co', 'inc', 'llc'])
    if is_short_and_ambiguous(name1) or is_short_and_ambiguous(name2):
        flags.add("Ambiguous Short Name")

    uncertainty_keywords = [
        "difficult to ascertain", "unclear", "ambiguous", "without further information",
        "requires more context", "potentially", "could be", "may be", "possibly",
        "depends on", "cannot definitively determine", "insufficient information",
        "fuzzy match", "appears to be",
        "likely but not certain", "it is not possible to determine",
        "no conclusive information", "uncertain"
    ]
    if any(keyword in llm_explanation.lower() for keyword in uncertainty_keywords):
        flags.add("LLM Unsure")

    return ", ".join(sorted(list(flags)))


def load_all_models_and_data():
    global sentence_bert_model, annoy_index, unique_customer_names, unique_name_to_embedding

    print("Loading Sentence-BERT model...", flush=True)
    try:
        sentence_bert_model = SentenceTransformer(LOCAL_MODEL_PATH)
        print("Sentence-BERT model loaded.", flush=True)
    except Exception as e:
        print(f"Failed to load Sentence-BERT model: {e}", flush=True)
        return False

    print("Loading unique customer names and embeddings map from CSV...", flush=True)
    if not os.path.exists(EMBEDDINGS_CSV_PATH):
        print(
            f"Error: Embeddings CSV not found at {EMBEDDINGS_CSV_PATH}. Please ensure your previous script generated it using ALL unique customer names.",
            flush=True)
        return False
    try:
        df_embeddings = pd.read_csv(EMBEDDINGS_CSV_PATH)
        unique_customer_names = df_embeddings['CustomerName'].tolist()
        df_embeddings['Embedding_Vector'] = df_embeddings['Embedding_Vector'].apply(
            lambda x: np.array(list(map(float, x.strip('[]').split(','))))
        )
        unique_name_to_embedding = pd.Series(
            df_embeddings.Embedding_Vector.values, index=df_embeddings.CustomerName
        ).to_dict()
        print(f"Loaded {len(unique_customer_names)} unique names and their embeddings.", flush=True)
    except Exception as e:
        print(f"Failed to load unique names or embeddings from CSV: {e}", flush=True)
        return False

    print("Loading Annoy index...", flush=True)
    try:
        annoy_index = AnnoyIndex(EMBEDDING_DIM, ANNOY_METRIC)
        print(f"AnnoyIndex object created with dimension {EMBEDDING_DIM} and metric '{ANNOY_METRIC}'.", flush=True)
        if annoy_index is None:
            print("ERROR: AnnoyIndex constructor returned None unexpectedly.", flush=True)
            return False
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize AnnoyIndex object: {e}", flush=True)
        return False

    if not os.path.exists(ANNOY_INDEX_FILE_PATH):
        print(
            f"Error: Annoy index file not found at {ANNOY_INDEX_FILE_PATH}. Please ensure your previous script generated it using ALL unique customer names.",
            flush=True)
        return False
    try:
        annoy_index.load(ANNOY_INDEX_FILE_PATH)
        print("Annoy index loaded.", flush=True)
    except Exception as e:
        print(f"Failed to load Annoy index: {e}", flush=True)
        if annoy_index is None:
            print("Reason for load failure: annoy_index object was None.", flush=True)
        return False

    return True


async def fill_customer_rollups_with_llm_and_annoy_async():
    global llm_decision_cache, llm_decision_count

    if not load_all_models_and_data():
        print("Failed to load necessary components. Exiting.", flush=True)
        return

    print("\n--- Starting Customer Rollup Filling Process ---", flush=True)

    try:
        df = pd.read_excel(INPUT_EXCEL_FILE)
        if COMPANY_NAME_COLUMN not in df.columns or \
                CUSTOMER_CODE_COLUMN not in df.columns or \
                CUSTOMER_ROLLUP_COLUMN not in df.columns or \
                CUSTOMER_PARENT_COLUMN not in df.columns:
            print(
                f"Error: Required columns ('{COMPANY_NAME_COLUMN}', '{CUSTOMER_CODE_COLUMN}', '{CUSTOMER_ROLLUP_COLUMN}', '{CUSTOMER_PARENT_COLUMN}') not found in '{INPUT_EXCEL_FILE}'.",
                flush=True)
            print("Please ensure your input Excel file has these exact column headers.", flush=True)
            return
    except FileNotFoundError:
        print(f"Error: Input Excel file not found at {INPUT_EXCEL_FILE}.", flush=True)
        print(
            "Please ensure you've run the initial script to create 'customerlistwrollupinitialized.xlsx' and that its path is correct.",
            flush=True)
        return
    except Exception as e:
        print(f"Error loading input Excel file: {e}", flush=True)
        return

    df[FLAGGED_COLUMN] = df[CUSTOMER_PARENT_COLUMN].apply(lambda x: False if pd.notna(x) and x != '' else np.nan)

    unlabeled_indices_full = df[
        (df[CUSTOMER_ROLLUP_COLUMN].isnull() | (df[CUSTOMER_ROLLUP_COLUMN] == ''))
    ].index.tolist()

    processed_companies_count = load_rollup_checkpoint()

    # --- Determine indices to process this run ---
    if processed_companies_count >= len(unlabeled_indices_full):
        print("All unlabeled items appear to have been processed according to checkpoint. Proceeding to final save.",
              flush=True)
        indices_to_process_this_run = []  # Nothing to process
    else:
        TEST_RUN_LIMIT = float('inf')  # Change this to 100 for a test run
        if len(unlabeled_indices_full) - processed_companies_count > TEST_RUN_LIMIT:
            print(f"--- Limiting processing to next {TEST_RUN_LIMIT} companies for this run. ---", flush=True)
            indices_to_process_this_run = unlabeled_indices_full[
                                          processed_companies_count: processed_companies_count + TEST_RUN_LIMIT]
        else:
            indices_to_process_this_run = unlabeled_indices_full[processed_companies_count:]

        print(
            f"Resuming processing from company {processed_companies_count + 1}. Remaining: {len(indices_to_process_this_run)} companies in this run.",
            flush=True)
    # --- End of MODIFICATION ---

    llm_decision_count = 0  # This should track new LLM calls in this session
    total_candidate_pairs = 0
    cache_hits = 0

    name_to_df_indices = df.groupby(COMPANY_NAME_COLUMN).apply(lambda x: x.index.tolist(),
                                                               include_groups=False).to_dict()

    companies_processed_this_session = 0  # Initialize for new processing

    try:
        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(CONCURRENT_API_CALLS_LIMIT)

            tasks = []

            for i, original_df_index in enumerate(indices_to_process_this_run):
                row = df.loc[original_df_index]

                # Re-check rollup status - important if async results can update other rows later in the same batch
                if pd.notna(df.loc[original_df_index, CUSTOMER_ROLLUP_COLUMN]) and df.loc[
                    original_df_index, CUSTOMER_ROLLUP_COLUMN] != '':
                    continue  # Skip to next index, it's already done.

                current_customer_code = row[CUSTOMER_CODE_COLUMN]
                current_customer_name = row[COMPANY_NAME_COLUMN]

                current_embedding = unique_name_to_embedding.get(current_customer_name)
                if current_embedding is None:
                    df.loc[original_df_index, CUSTOMER_ROLLUP_COLUMN] = current_customer_code
                    df.loc[original_df_index, FLAGGED_COLUMN] = True
                    print(
                        f"Warning: No embedding found for '{current_customer_name}'. Assigned own CustomerCode as rollup and flagged. (Non-async path)",
                        flush=True)
                    companies_processed_this_session += 1  # Increment for synchronous path
                    if (
                            processed_companies_count + companies_processed_this_session) % CHECKPOINT_INTERVAL_COMPANIES == 0:
                        save_rollup_checkpoint(processed_companies_count + companies_processed_this_session)
                    continue

                neighbor_ids, distances = annoy_index.get_nns_by_vector(
                    current_embedding, n=20, include_distances=True
                )

                llm_candidates_for_current = []
                assigned_rollup_via_cache = False

                for neighbor_id, dist in zip(neighbor_ids, distances):
                    candidate_name = unique_customer_names[neighbor_id]
                    cos_similarity = 1 - (dist ** 2 / 2)

                    if current_customer_name == candidate_name:
                        continue

                    if cos_similarity < ANNOY_CANDIDATE_SIMILARITY_THRESHOLD:
                        continue

                    candidate_df_indices = name_to_df_indices.get(candidate_name, [])
                    if not candidate_df_indices:
                        continue

                    candidate_original_df_index = candidate_df_indices[0]  # Get first index for candidate
                    candidate_row = df.loc[candidate_original_df_index]
                    candidate_rollup = candidate_row[CUSTOMER_ROLLUP_COLUMN]
                    candidate_customer_code = candidate_row[
                        CUSTOMER_CODE_COLUMN]  # Pass candidate's code for leader logic

                    is_candidate_rollup_active = pd.notna(candidate_rollup) and candidate_rollup != ''

                    pair_key = frozenset({current_customer_name, candidate_name})

                    if pair_key in llm_decision_cache:
                        is_same, explanation = llm_decision_cache[pair_key]
                        cache_hits += 1
                        # If a cached decision says they are the same:
                        if is_same:
                            # If candidate has an active rollup, use it
                            if is_candidate_rollup_active:
                                df.loc[original_df_index, CUSTOMER_ROLLUP_COLUMN] = candidate_rollup
                                df.loc[original_df_index, FLAGGED_COLUMN] = True
                                print(
                                    f"  Cached match found for '{current_customer_name}' to '{candidate_name}'. Assigned '{candidate_rollup}'.",
                                    flush=True)
                                assigned_rollup_via_cache = True
                                companies_processed_this_session += 1
                                if (
                                        processed_companies_count + companies_processed_this_session) % CHECKPOINT_INTERVAL_COMPANIES == 0:
                                    save_rollup_checkpoint(processed_companies_count + companies_processed_this_session)
                                break  # Found a solid rollup, no need for more candidates for this customer
                            else:
                                # If candidate DOES NOT have an active rollup but they are cached as same
                                # This is where we apply the leader logic if both are unassigned.
                                # Prepare for async to handle this.
                                llm_candidates_for_current.append({
                                    'pair_key': pair_key,
                                    'current_name': current_customer_name,
                                    'candidate_name': candidate_name,
                                    'is_candidate_rollup_active': is_candidate_rollup_active,  # This will be False
                                    'candidate_rollup': candidate_rollup,  # This will be NaN/empty
                                    'candidate_customer_code': candidate_customer_code,
                                    # IMPORTANT: Pass candidate's code
                                    'original_df_index': original_df_index,
                                    'customer_code': current_customer_code,
                                    'customer_name': current_customer_name
                                })
                                total_candidate_pairs += 1  # Still count as potential LLM related interaction
                        # If cached as NOT same, just continue to next candidate
                    else:
                        # No cache hit, needs LLM call
                        llm_candidates_for_current.append({
                            'pair_key': pair_key,
                            'current_name': current_customer_name,
                            'candidate_name': candidate_name,
                            'is_candidate_rollup_active': is_candidate_rollup_active,
                            'candidate_rollup': candidate_rollup,
                            'candidate_customer_code': candidate_customer_code,  # IMPORTANT: Pass candidate's code
                            'original_df_index': original_df_index,
                            'customer_code': current_customer_code,
                            'customer_name': current_customer_name
                        })
                        total_candidate_pairs += 1

                if not assigned_rollup_via_cache:
                    if llm_candidates_for_current:
                        tasks.append(process_customer_async(
                            session, semaphore, original_df_index, current_customer_code, current_customer_name,
                            llm_candidates_for_current
                        ))
                    elif pd.isna(df.loc[original_df_index, CUSTOMER_ROLLUP_COLUMN]) or df.loc[
                        original_df_index, CUSTOMER_ROLLUP_COLUMN] == '':
                        df.loc[original_df_index, CUSTOMER_ROLLUP_COLUMN] = current_customer_code
                        df.loc[original_df_index, FLAGGED_COLUMN] = True
                        print(
                            f"  No suitable candidates or cached match found for '{current_customer_name}'. Assigned own code. (Non-async path)",
                            flush=True)
                        companies_processed_this_session += 1
                        if (
                                processed_companies_count + companies_processed_this_session) % CHECKPOINT_INTERVAL_COMPANIES == 0:
                            save_rollup_checkpoint(processed_companies_count + companies_processed_this_session)

            print(f"Initiating {len(tasks)} asynchronous tasks for LLM calls...", flush=True)
            async_results = await asyncio.gather(*tasks, return_exceptions=True)

            print("Applying results from asynchronous tasks to DataFrame...", flush=True)
            for result in async_results:
                if isinstance(result, Exception):
                    # In case of an exception during an async task, ensure the original row is handled.
                    # This means it will fall back to its own code and be flagged.
                    print(f"  Error in async task: {result}", flush=True)
                else:
                    original_df_index = result['original_df_index']
                    current_customer_name = result['current_customer_name']
                    current_customer_code = result['current_customer_code']
                    assigned_rollup_value = result['assigned_rollup_value']
                    flagged_status = result['flagged_status']

                    # Ensure we only update if it hasn't been set by another process or a synchronous path
                    # (e.g., if one of the companies in an async pair was processed by another task first)
                    if pd.isna(df.loc[original_df_index, CUSTOMER_ROLLUP_COLUMN]) or df.loc[
                        original_df_index, CUSTOMER_ROLLUP_COLUMN] == '':
                        df.loc[original_df_index, CUSTOMER_ROLLUP_COLUMN] = assigned_rollup_value
                        df.loc[original_df_index, FLAGGED_COLUMN] = flagged_status
                        print(
                            f"  Finalized rollup for '{current_customer_name}' ({current_customer_code}) to '{assigned_rollup_value}'. Flagged: {flagged_status}",
                            flush=True)
                        companies_processed_this_session += 1
                        # Checkpoint logic for async results applied after the task completes
                        if (
                                processed_companies_count + companies_processed_this_session) % CHECKPOINT_INTERVAL_COMPANIES == 0:
                            save_rollup_checkpoint(processed_companies_count + companies_processed_this_session)
                    else:
                        print(
                            f"  Rollup for '{current_customer_name}' ({current_customer_code}) already set by another process or cache hit. Skipping update.",
                            flush=True)

    finally:
        # Save the final checkpoint, representing total progress, regardless of Excel save success
        save_rollup_checkpoint(processed_companies_count + companies_processed_this_session)

    # Fill any remaining NaN in 'Flagged' column with False
    df[FLAGGED_COLUMN] = df[FLAGGED_COLUMN].fillna(False)

    print(
        f"\nFULL ASYNC RUN Completed. CustomerRollup filling and flagging for {companies_processed_this_session} companies this session.",
        flush=True)
    print(f"Total candidate pairs considered by Annoy: {total_candidate_pairs}", flush=True)
    print(f"Total LLM API calls made: {llm_decision_count}", flush=True)
    print(f"Total LLM cache hits: {cache_hits}", flush=True)

    print(f"Saving results to '{OUTPUT_EXCEL_FILE}'...", flush=True)

    # --- Start of Excel Save with Retry Logic (same as before) ---
    MAX_EXCEL_SAVE_RETRIES = 5
    EXCEL_RETRY_DELAY_SECONDS = 10  # Wait 10 seconds between attempts

    for attempt in range(1, MAX_EXCEL_SAVE_RETRIES + 1):
        try:
            df.to_excel(OUTPUT_EXCEL_FILE, index=False)
            print(f"CustomerRollup filling and flagging complete! Results saved to '{OUTPUT_EXCEL_FILE}'.", flush=True)
            break  # Exit loop if save is successful
        except PermissionError as e:
            if attempt < MAX_EXCEL_SAVE_RETRIES:
                print(
                    f"Permission Error: '{OUTPUT_EXCEL_FILE}' is open. Please close it. Retrying in {EXCEL_RETRY_DELAY_SECONDS} seconds... (Attempt {attempt}/{MAX_EXCEL_SAVE_RETRIES})",
                    flush=True)
                time.sleep(EXCEL_RETRY_DELAY_SECONDS)
            else:
                print(
                    f"Final attempt failed: Permission Error: '{OUTPUT_EXCEL_FILE}' is still open after {MAX_EXCEL_SAVE_RETRIES} attempts. Please close it and manually save the DataFrame.",
                    flush=True)
                # Optionally, you could save to a temp file here if critical to not lose data
                # df.to_excel(OUTPUT_EXCEL_FILE.replace(".xlsx", "_FAILED_SAVE.xlsx"), index=False)
                # print(f"Data saved to a temporary file: {OUTPUT_EXCEL_FILE.replace('.xlsx', '_FAILED_SAVE.xlsx')}", flush=True)
        except Exception as e:
            print(f"An unexpected error occurred while saving to Excel: {e}", flush=True)
            break  # Exit loop for other unexpected errors
    # --- End of Excel Save with Retry Logic ---


# MODIFIED: process_customer_async signature to take current_customer_code and name directly
async def process_customer_async(session, semaphore, original_df_index, current_customer_code, current_customer_name,
                                 llm_candidates):
    global llm_decision_cache, llm_decision_count

    best_match_rollup = None
    found_rollup_match_with_active_parent = False

    # Store potential 'same' companies that don't have an active rollup yet
    # This list will be used if no active parent is found.
    unassigned_same_companies_codes = []

    # Add the current customer's code to this list initially, as it's also unassigned.
    unassigned_same_companies_codes.append(current_customer_code)

    for candidate_info in llm_candidates:
        pair_key = candidate_info['pair_key']
        candidate_name = candidate_info['candidate_name']
        is_candidate_rollup_active = candidate_info['is_candidate_rollup_active']
        candidate_rollup = candidate_info['candidate_rollup']
        candidate_customer_code = candidate_info['candidate_customer_code']  # Retrieve candidate's code

        is_same = False
        explanation = "Not processed by LLM or cache."

        # Check cache first for this specific pair
        if pair_key in llm_decision_cache:
            is_same, explanation = llm_decision_cache[pair_key]
            # Cache hit count is already handled in the main loop for this branch
        else:
            # If not in cache, make the LLM call
            async with semaphore:
                is_same, explanation = await ask_gemini_if_same_company_async(session, current_customer_name,
                                                                              candidate_name)
            llm_decision_cache[pair_key] = (is_same, explanation)

        if is_same:
            if is_candidate_rollup_active:
                best_match_rollup = candidate_rollup
                found_rollup_match_with_active_parent = True
                print(
                    f"    LLM confirmed '{current_customer_name}' and '{candidate_name}' are same. Using candidate's active rollup: '{candidate_rollup}'.",
                    flush=True)
                break  # Found a definitive active rollup, no need to check further candidates
            else:
                # If they are the same, but the candidate itself doesn't have an active rollup,
                # add its code to the list for potential arbitrary assignment later.
                unassigned_same_companies_codes.append(candidate_customer_code)
                print(
                    f"    LLM confirmed '{current_customer_name}' and '{candidate_name}' are same, but candidate '{candidate_name}' has no active rollup. Adding '{candidate_customer_code}' to pool.",
                    flush=True)
        else:
            print(
                f"    LLM determined '{current_customer_name}' and '{candidate_name}' are DIFFERENT. Explanation: {explanation}",
                flush=True)

    assigned_rollup_value = None
    flagged_status = True  # Assume flagged if LLM involved or complex assignment

    if found_rollup_match_with_active_parent:
        assigned_rollup_value = best_match_rollup
    else:
        # No active parent found among candidates or via cache.
        # Now, choose a deterministic rollup from the unassigned_same_companies_codes pool.
        # This resolves the reciprocal assignment issue by ensuring they pick the same leader.
        if unassigned_same_companies_codes:
            # Sort codes and pick the numerically/lexicographically smallest
            # This ensures if A is compared to B, and B to A, they both pick the same leader.
            assigned_rollup_value = min(unassigned_same_companies_codes)
            print(
                f"    No active parent found for '{current_customer_name}'. Assigned deterministic rollup: '{assigned_rollup_value}' from pool: {unassigned_same_companies_codes}.",
                flush=True)
        else:
            # If no candidates confirmed as 'same' (neither active nor unassigned),
            # then the current company becomes its own rollup.
            assigned_rollup_value = current_customer_code
            print(
                f"    No suitable 'same' candidates found for '{current_customer_name}'. Assigned own code: '{current_customer_code}'.",
                flush=True)

    # Return the result for the main loop to apply
    return {
        'original_df_index': original_df_index,
        'current_customer_code': current_customer_code,
        'current_customer_name': current_customer_name,
        'assigned_rollup_value': assigned_rollup_value,
        'flagged_status': flagged_status
    }


# --- New function to regenerate Excel from cache ---
# This function MUST be defined BEFORE the if __name__ == "__main__": block
# --- New function to regenerate Excel from cache ---
async def generate_excel_from_cache():
    global llm_decision_cache, llm_decision_count

    if not load_all_models_and_data():
        print("Failed to load necessary components (models, embeddings, annoy index). Exiting.", flush=True)
        return

    print("\n--- Starting Excel Regeneration from Cache ---", flush=True)

    try:
        df = pd.read_excel(INPUT_EXCEL_FILE)
        if COMPANY_NAME_COLUMN not in df.columns or \
                CUSTOMER_CODE_COLUMN not in df.columns or \
                CUSTOMER_ROLLUP_COLUMN not in df.columns or \
                CUSTOMER_PARENT_COLUMN not in df.columns:
            print(
                f"Error: Required columns ('{COMPANY_NAME_COLUMN}', '{CUSTOMER_CODE_COLUMN}', '{CUSTOMER_ROLLUP_COLUMN}', '{CUSTOMER_PARENT_COLUMN}') not found in '{INPUT_EXCEL_FILE}'.",
                flush=True)
            print("Please ensure your input Excel file has these exact column headers.", flush=True)
            return
    except FileNotFoundError:
        print(f"Error: Input Excel file not found at {INPUT_EXCEL_FILE}.", flush=True)
        return
    except Exception as e:
        print(f"Error loading input Excel file: {e}", flush=True)
        return

    # Initialize Flagged column - start all as False
    df[FLAGGED_COLUMN] = False

    # --- CRITICAL CHANGE 1: Initialize CUSTOMER_ROLLUP_COLUMN based on CUSTOMER_PARENT_COLUMN or self ---
    df[CUSTOMER_ROLLUP_COLUMN] = df.apply(
        lambda row: row[CUSTOMER_PARENT_COLUMN] if pd.notna(row[CUSTOMER_PARENT_COLUMN]) and row[
            CUSTOMER_PARENT_COLUMN] != ''
        else row[CUSTOMER_CODE_COLUMN], axis=1
    )
    # If a row has a CustomerParent, its initial rollup is that parent. Otherwise, it's itself.
    # This ensures explicit parents are respected from the very start.

    # Load the full cache from the previous run
    processed_codes_count_from_checkpoint = load_rollup_checkpoint()  # This populates llm_decision_cache
    if not llm_decision_cache:
        print(
            "Warning: LLM decision cache is empty. Regenerating Excel will primarily use self-assignment and only flag based on basic name properties.",
            flush=True)
    print(f"Loaded {len(llm_decision_cache)} LLM decisions from cache.", flush=True)

    name_to_df_indices = df.groupby(COMPANY_NAME_COLUMN).apply(lambda x: x.index.tolist(),
                                                               include_groups=False).to_dict()

    processed_companies_count_in_regen = 0

    # Iterate over ALL rows to ensure all are considered for rollup and flagging
    for original_df_index in df.index:
        row = df.loc[original_df_index]
        current_customer_code = row[CUSTOMER_CODE_COLUMN]
        current_customer_name = row[COMPANY_NAME_COLUMN]

        # --- Handle rows that already have a CustomerParent as per initial data ---
        if pd.notna(row[CUSTOMER_PARENT_COLUMN]) and row[CUSTOMER_PARENT_COLUMN] != '':
            # Rollup already set to CustomerParent above. Just mark as not flagged.
            df.loc[original_df_index, FLAGGED_COLUMN] = False
            processed_companies_count_in_regen += 1
            continue  # Skip further processing for explicitly parented rows

        # If it doesn't have a CustomerParent (or it's NaN/empty), proceed with finding rollup and flagging
        current_embedding = unique_name_to_embedding.get(current_customer_name)
        if current_embedding is None:
            # Rollup is already defaulted to own code from initial setup. Just flag.
            df.loc[original_df_index, FLAGGED_COLUMN] = True  # Flagged if no embedding
            print(
                f"Warning: No embedding found for '{current_customer_name}'. Assigned own CustomerCode as rollup and flagged during regeneration.",
                flush=True)
            processed_companies_count_in_regen += 1
            continue

        neighbor_ids, distances = annoy_index.get_nns_by_vector(
            current_embedding, n=20, include_distances=True
        )

        found_rollup_match_with_active_parent = False
        potential_rollup_candidate = current_customer_code  # Default to own code initially
        llm_explanation_for_flagging = ""
        is_llm_same_decision = False

        # Store potential 'same' companies that don't have an active rollup yet
        unassigned_same_companies_codes_in_regen = [current_customer_code]

        for neighbor_id, dist in zip(neighbor_ids, distances):
            candidate_name = unique_customer_names[neighbor_id]
            cos_similarity = 1 - (dist ** 2 / 2)

            if current_customer_name == candidate_name:
                continue

            if cos_similarity < ANNOY_CANDIDATE_SIMILARITY_THRESHOLD:
                continue

            candidate_df_indices = name_to_df_indices.get(candidate_name, [])
            if not candidate_df_indices:
                continue

            candidate_original_df_index = candidate_df_indices[0]
            candidate_row = df.loc[candidate_original_df_index]
            candidate_rollup = candidate_row[
                CUSTOMER_ROLLUP_COLUMN]  # Use the current state of DF, which now prioritizes CustomerParent
            candidate_customer_code = candidate_row[CUSTOMER_CODE_COLUMN]

            is_candidate_rollup_active = pd.notna(candidate_rollup) and candidate_rollup != ''

            pair_key = frozenset({current_customer_name, candidate_name})

            if pair_key in llm_decision_cache:
                is_same, explanation = llm_decision_cache[pair_key]
                llm_explanation_for_flagging = explanation
                is_llm_same_decision = is_same
                if is_same:
                    if is_candidate_rollup_active:
                        potential_rollup_candidate = candidate_rollup
                        found_rollup_match_with_active_parent = True
                        print(
                            f"  Regen: Cached match found for '{current_customer_name}' to '{candidate_name}'. Assigned '{candidate_rollup}'.",
                            flush=True)
                        break
                    else:
                        unassigned_same_companies_codes_in_regen.append(candidate_customer_code)

        if found_rollup_match_with_active_parent:
            df.loc[original_df_index, CUSTOMER_ROLLUP_COLUMN] = potential_rollup_candidate
            df.loc[original_df_index, FLAGGED_COLUMN] = True
        else:
            if unassigned_same_companies_codes_in_regen:
                df.loc[original_df_index, CUSTOMER_ROLLUP_COLUMN] = min(unassigned_same_companies_codes_in_regen)
                print(
                    f"  Regen: No active parent found for '{current_customer_name}'. Assigned deterministic rollup: '{df.loc[original_df_index, CUSTOMER_ROLLUP_COLUMN]}'.",
                    flush=True)
            else:
                df.loc[original_df_index, CUSTOMER_ROLLUP_COLUMN] = current_customer_code
                print(
                    f"  Regen: No suitable 'same' candidates from cache for '{current_customer_name}'. Assigned own code: '{current_customer_code}'.",
                    flush=True)

            flags = analyze_for_flagging(current_customer_name, "", llm_explanation_for_flagging, is_llm_same_decision)
            if flags:
                df.loc[original_df_index, FLAGGED_COLUMN] = True

        processed_companies_count_in_regen += 1

    # --- Start of Rollup Chain Resolution ---
    print("\n--- Resolving Rollup Chains for Consistency ---", flush=True)

    # 1. Create a dictionary to map CustomerCode to its assigned CustomerRollup
    current_rollup_assignments = {
        row[CUSTOMER_CODE_COLUMN]: row[CUSTOMER_ROLLUP_COLUMN]
        for index, row in df.iterrows()
        if pd.notna(row[CUSTOMER_ROLLUP_COLUMN]) and row[CUSTOMER_ROLLUP_COLUMN] != ''
    }

    # 2. Iteratively resolve each rollup to its ultimate parent
    resolved_rollups = {}  # Stores the final, flattened rollup for each customer code

    for customer_code in df[CUSTOMER_CODE_COLUMN].unique():  # Iterate through all unique customer codes
        current_trace_code = customer_code
        path = [current_trace_code]  # To detect cycles

        # Keep tracing as long as the current code has an assigned rollup *within our dataset*
        # and we haven't detected a cycle.
        # This while loop condition prioritizes explicit CustomerParent from the data.
        # It ensures that if a CustomerCode points to its CustomerParent, we don't try to trace
        # further in this chain, as CustomerParent is considered a defined "root" for that specific company.
        while (current_trace_code in current_rollup_assignments and
               current_rollup_assignments[current_trace_code] != current_trace_code and
               (current_trace_code != df.loc[df[CUSTOMER_CODE_COLUMN] == customer_code, CUSTOMER_PARENT_COLUMN].iloc[
                   0] if (df[CUSTOMER_CODE_COLUMN] == customer_code).any() and pd.notna(
                   df.loc[df[CUSTOMER_CODE_COLUMN] == customer_code, CUSTOMER_PARENT_COLUMN].iloc[0]) and
                         df.loc[df[CUSTOMER_CODE_COLUMN] == customer_code, CUSTOMER_PARENT_COLUMN].iloc[
                             0] != '' else True) and
               current_rollup_assignments[current_trace_code] not in path):

            next_rollup = current_rollup_assignments[current_trace_code]

            # If the next rollup is already resolved, use its resolution directly
            if next_rollup in resolved_rollups:
                current_trace_code = resolved_rollups[next_rollup]
                break

            path.append(next_rollup)
            current_trace_code = next_rollup

        # The 'current_trace_code' is now the ultimate parent for this chain (or itself if no chain)
        resolved_rollups[customer_code] = current_trace_code

        # Only print if a change occurred and it's not a self-assignment to start
        if customer_code != resolved_rollups[customer_code] and resolved_rollups[
            customer_code] != current_rollup_assignments.get(customer_code, customer_code):
            print(f"  Resolved rollup for {customer_code} to {resolved_rollups[customer_code]}", flush=True)

    # 3. Apply the resolved rollups back to the DataFrame
    df[CUSTOMER_ROLLUP_COLUMN] = df[CUSTOMER_CODE_COLUMN].map(resolved_rollups)

    print("--- Rollup Chain Resolution Complete ---", flush=True)
    # --- End of Rollup Chain Resolution ---

    # Final fillna(False) for any remaining NaN, although should be caught by initialization or logic.
    df[FLAGGED_COLUMN] = df[FLAGGED_COLUMN].fillna(False)

    print(f"\nExcel Regeneration Completed. Processed {processed_companies_count_in_regen} companies.", flush=True)

    # --- Start of Excel Save with Retry Logic ---
    MAX_EXCEL_SAVE_RETRIES = 5
    EXCEL_RETRY_DELAY_SECONDS = 10  # Wait 10 seconds between attempts

    for attempt in range(1, MAX_EXCEL_SAVE_RETRIES + 1):
        try:
            df.to_excel(OUTPUT_EXCEL_FILE, index=False)
            print(f"Results saved to '{OUTPUT_EXCEL_FILE}'.", flush=True)
            break  # Exit loop if save is successful
        except PermissionError as e:
            if attempt < MAX_EXCEL_SAVE_RETRIES:
                print(
                    f"Permission Error: '{OUTPUT_EXCEL_FILE}' is open. Please close it. Retrying in {EXCEL_RETRY_DELAY_SECONDS} seconds... (Attempt {attempt}/{MAX_EXCEL_SAVE_RETRIES})",
                    flush=True)
                time.sleep(EXCEL_RETRY_DELAY_SECONDS)
            else:
                print(
                    f"Final attempt failed: Permission Error: '{OUTPUT_EXCEL_FILE}' is still open after {MAX_EXCEL_SAVE_RETRIES} attempts. Please close it and manually save the DataFrame.",
                    flush=True)
                # Optionally, you could save to a temp file here if critical to not lose data
                # df.to_excel(OUTPUT_EXCEL_FILE.replace(".xlsx", "_FAILED_SAVE.xlsx"), index=False)
                # print(f"Data saved to a temporary file: {OUTPUT_EXCEL_FILE.replace('.xlsx', '_FAILED_SAVE.xlsx')}", flush=True)
        except Exception as e:
            print(f"An unexpected error occurred while saving to Excel: {e}", flush=True)
            break  # Exit loop for other unexpected errors
    # --- End of Excel Save with Retry Logic ---


# MODIFIED: process_customer_async signature to take current_customer_code and name directly
async def process_customer_async(session, semaphore, original_df_index, current_customer_code, current_customer_name,
                                 llm_candidates):
    global llm_decision_cache, llm_decision_count

    best_match_rollup = None
    found_rollup_match_with_active_parent = False

    # Store potential 'same' companies that don't have an active rollup yet
    # This list will be used if no active parent is found.
    unassigned_same_companies_codes = []

    # Add the current customer's code to this list initially, as it's also unassigned.
    unassigned_same_companies_codes.append(current_customer_code)

    for candidate_info in llm_candidates:
        pair_key = candidate_info['pair_key']
        candidate_name = candidate_info['candidate_name']
        is_candidate_rollup_active = candidate_info['is_candidate_rollup_active']
        candidate_rollup = candidate_info['candidate_rollup']
        candidate_customer_code = candidate_info['candidate_customer_code']  # Retrieve candidate's code

        is_same = False
        explanation = "Not processed by LLM or cache."

        # Check cache first for this specific pair
        if pair_key in llm_decision_cache:
            is_same, explanation = llm_decision_cache[pair_key]
            # Cache hit count is already handled in the main loop for this branch
        else:
            # If not in cache, make the LLM call
            async with semaphore:
                is_same, explanation = await ask_gemini_if_same_company_async(session, current_customer_name,
                                                                              candidate_name)
            llm_decision_cache[pair_key] = (is_same, explanation)

        if is_same:
            if is_candidate_rollup_active:
                best_match_rollup = candidate_rollup
                found_rollup_match_with_active_parent = True
                print(
                    f"    LLM confirmed '{current_customer_name}' and '{candidate_name}' are same. Using candidate's active rollup: '{candidate_rollup}'.",
                    flush=True)
                break  # Found a definitive active rollup, no need to check further candidates
            else:
                # If they are the same, but the candidate itself doesn't have an active rollup,
                # add its code to the list for potential arbitrary assignment later.
                unassigned_same_companies_codes.append(candidate_customer_code)
                print(
                    f"    LLM confirmed '{current_customer_name}' and '{candidate_name}' are same, but candidate '{candidate_name}' has no active rollup. Adding '{candidate_customer_code}' to pool.",
                    flush=True)
        else:
            print(
                f"    LLM determined '{current_customer_name}' and '{candidate_name}' are DIFFERENT. Explanation: {explanation}",
                flush=True)

    assigned_rollup_value = None
    flagged_status = True  # Assume flagged if LLM involved or complex assignment

    if found_rollup_match_with_active_parent:
        assigned_rollup_value = best_match_rollup
    else:
        # No active parent found among candidates or via cache.
        # Now, choose a deterministic rollup from the unassigned_same_companies_codes pool.
        # This resolves the reciprocal assignment issue by ensuring they pick the same leader.
        if unassigned_same_companies_codes:
            # Sort codes and pick the numerically/lexicographically smallest
            # This ensures if A is compared to B, and B to A, they both pick the same leader.
            assigned_rollup_value = min(unassigned_same_companies_codes)
            print(
                f"    No active parent found for '{current_customer_name}'. Assigned deterministic rollup: '{assigned_rollup_value}' from pool: {unassigned_same_companies_codes}.",
                flush=True)
        else:
            # If no candidates confirmed as 'same' (neither active nor unassigned),
            # then the current company becomes its own rollup.
            assigned_rollup_value = current_customer_code
            print(
                f"    No suitable 'same' candidates found for '{current_customer_name}'. Assigned own code: '{current_customer_code}'.",
                flush=True)

    # Return the result for the main loop to apply
    return {
        'original_df_index': original_df_index,
        'current_customer_code': current_customer_code,
        'current_customer_name': current_customer_name,
        'assigned_rollup_value': assigned_rollup_value,
        'flagged_status': flagged_status
    }


# --- Main execution block ---
if __name__ == "__main__":
    try:
        from fuzzywuzzy import fuzz
    except ImportError:
        print("Warning: fuzzywuzzy not found. Install with 'pip install fuzzywuzzy python-Levenshtein'.", flush=True)


        class DummyFuzz:
            def ratio(self, s1, s2): return 0


        fuzz = DummyFuzz()

    # --- Choose which function to run ---
    # Uncomment one of the following lines:

    # For a full processing run (making LLM calls where needed, and updating checkpoints)
    #asyncio.run(fill_customer_rollups_with_llm_and_annoy_async())

    # For generating the Excel file ONLY from existing cache data (no new LLM calls)
    asyncio.run(generate_excel_from_cache())