import pandas as pd
import numpy as np
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer
import os
import re
import time
from fuzzywuzzy import fuzz
import json
import socket
import asyncio
import aiohttp
import random
import signal

# --- Global flag for graceful exit ---
should_exit = False


def signal_handler(sig, frame):
    global should_exit
    print("\nCtrl+C detected! Initiating graceful shutdown. Please wait...", flush=True)
    should_exit = True


# --- Configuration ---
MASTER_EXCEL_FILE = r"C:\Users\JackLindgren\OneDrive - Burwell Material Handling\Desktop\Data\Completed\Reducedcustomerlistwrollup_final_cleaned_rollup.xlsx"
NAVISION_EXCEL_FILE = r"C:\Users\JackLindgren\OneDrive - Burwell Material Handling\Desktop\Data\Navision_Customers_Removed_Migrated.xlsx"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DEDUP_EXCEL_FILE = os.path.join(SCRIPT_DIR, "Navision_Deduplicated.xlsx")

MASTER_COMPANY_NAME_COLUMN = 'CustomerName'
MASTER_CUSTOMER_CODE_COLUMN = 'CustomerCode'
NAVISION_COMPANY_NAME_COLUMN = 'Name'

IS_DUPLICATE_COLUMN = 'IsDuplicate'
MATCHED_MASTER_COMPANY_NAME_COLUMN = 'MatchedMasterCompanyName'
MATCHED_MASTER_CUSTOMER_CODE_COLUMN = 'MatchedMasterCustomerCode'
LLM_EXPLANATION_COLUMN = 'LLMExplanation'
FLAGGED_FOR_REVIEW_COLUMN = 'FlaggedForReview'

LOCAL_MODEL_PATH = r"C:\LLm\mpnet"
ANNOY_INDEX_FILE_PATH = r"C:\Users\JackLindgren\PycharmProjects\PythonProject1\customer_annoy_index.ann"
EMBEDDINGS_CSV_PATH = r"C:\Users\JackLindgren\PycharmProjects\PythonProject1\unique_customer_embeddings.csv"
EMBEDDING_DIM = 768
ANNOY_METRIC = 'angular'

ANNOY_CANDIDATE_SIMILARITY_THRESHOLD = 0.5
FUZZY_MATCH_THRESHOLD_INITIAL = 95  # High threshold for quick, obvious matches *before* LLM

GEMINI_MODEL_NAME = "gemini-1.5-flash"
API_CALL_DELAY_SECONDS = 0.05
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 3
CONCURRENT_API_CALLS_LIMIT = 20  # Still relevant for async LLM calls

CHECKPOINT_DIR_DEDUP = os.path.join(SCRIPT_DIR, "dedup_checkpoints")
DEDUP_LLM_CACHE_FILE = os.path.join(CHECKPOINT_DIR_DEDUP, "dedup_llm_cache.json")
DEDUP_PROGRESS_FILE = os.path.join(CHECKPOINT_DIR_DEDUP, "dedup_progress.json")
CHECKPOINT_INTERVAL_COMPANIES = 50

# --- Global Models and Data (loaded once) ---
sentence_bert_model = None
annoy_index = None
master_unique_customer_names = []
master_name_to_code = {}

# --- Deduplication Specific Cache for LLM calls ---
dedup_llm_decision_cache = {}
dedup_llm_call_count = 0
dedup_llm_cache_hits = 0


# --- Helper Functions ---

def check_internet_connection(timeout=5):
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=timeout)
        return True
    except OSError:
        return False


def save_dedup_checkpoint(processed_navision_count):
    os.makedirs(CHECKPOINT_DIR_DEDUP, exist_ok=True)
    try:
        serializable_cache = {
            json.dumps(sorted(list(k))): v
            for k, v in dedup_llm_decision_cache.items()
        }
        with open(DEDUP_LLM_CACHE_FILE, 'w') as f:
            json.dump(serializable_cache, f, indent=4)

        with open(DEDUP_PROGRESS_FILE, 'w') as f:
            json.dump({"processed_navision_count": processed_navision_count}, f)

        print(f"\n--- Deduplication Checkpoint saved at {processed_navision_count} Navision companies processed. ---",
              flush=True)
    except Exception as e:
        print(f"Error saving deduplication checkpoint: {e}. Check directory permissions or disk space.", flush=True)


def load_dedup_checkpoint():
    global dedup_llm_decision_cache, dedup_llm_call_count
    processed_navision_count = 0

    if os.path.exists(DEDUP_LLM_CACHE_FILE) and os.path.exists(DEDUP_PROGRESS_FILE):
        try:
            with open(DEDUP_LLM_CACHE_FILE, 'r') as f:
                deserialized_cache = json.load(f)
                dedup_llm_decision_cache = {
                    frozenset(json.loads(k)): tuple(v) if isinstance(v, list) else v
                    for k, v in deserialized_cache.items()
                }

            with open(DEDUP_PROGRESS_FILE, 'r') as f:
                progress_data = json.load(f)
                processed_navision_count = progress_data.get("processed_navision_count", 0)

            print("\n--- Deduplication Checkpoint loaded successfully. Resuming from last state. ---", flush=True)
            dedup_llm_call_count = len(dedup_llm_decision_cache)
            return processed_navision_count
        except Exception as e:
            print(f"Error loading deduplication checkpoint, starting fresh: {e}", flush=True)
            dedup_llm_decision_cache.clear()
            dedup_llm_call_count = 0
            return 0
    return 0


async def ask_gemini_if_same_company_async(session, name1: str, name2: str) -> tuple[bool, str]:
    global dedup_llm_call_count, dedup_llm_cache_hits

    pair_key = frozenset({name1, name2})
    if pair_key in dedup_llm_decision_cache:
        dedup_llm_cache_hits += 1
        return dedup_llm_decision_cache[pair_key]

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY environment variable not set. LLM calls will fail.", flush=True)
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
        if should_exit:
            final_explanation = "Graceful exit initiated during API call."
            dedup_llm_decision_cache[pair_key] = (False, final_explanation)
            raise asyncio.CancelledError("Graceful exit requested.")

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

            dedup_llm_call_count += 1
            dedup_llm_decision_cache[pair_key] = (llm_decision, llm_explanation)
            return llm_decision, llm_explanation

        except aiohttp.ClientResponseError as e:
            retries += 1
            print(
                f"  Gemini API HTTP Error for '{name1}' vs '{name2}' (Retry {retries}/{MAX_RETRIES}): Status {e.status}, Message: {e.message}",
                flush=True)
            if e.status == 429:
                retry_delay = INITIAL_RETRY_DELAY * (2 ** retries) + random.uniform(0, 1)
                print(f"  Rate limit hit. Waiting {retry_delay:.2f} seconds before retrying...", flush=True)
                await asyncio.sleep(retry_delay)
            elif e.status == 503:
                retry_delay = INITIAL_RETRY_DELAY * (2 ** retries) + random.uniform(0, 1)
                print(f"  Service overloaded (503). Waiting {retry_delay:.2f} seconds before retrying...", flush=True)
                await asyncio.sleep(retry_delay)
            else:
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

    final_explanation = f"Failed after {MAX_RETRIES} retries due to Gemini API errors."
    dedup_llm_decision_cache[pair_key] = (False, final_explanation)
    return False, final_explanation


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

    if not is_llm_same:
        flags.add("LLM Determined Different")

    return ", ".join(sorted(list(flags)))


def load_all_models_and_data_for_deduplication():
    global sentence_bert_model, annoy_index, master_unique_customer_names, master_name_to_code

    print("Loading Sentence-BERT model...", flush=True)
    try:
        sentence_bert_model = SentenceTransformer(LOCAL_MODEL_PATH)
        detected_embedding_dim = sentence_bert_model.get_sentence_embedding_dimension()
        print(
            f"Sentence-BERT model loaded from '{LOCAL_MODEL_PATH}'. Detected embedding dimension: {detected_embedding_dim}",
            flush=True)

        if detected_embedding_dim != EMBEDDING_DIM:
            print(
                f"WARNING: Detected model embedding dimension ({detected_embedding_dim}) does not match configured EMBEDDING_DIM ({EMBEDDING_DIM}).",
                flush=True)
            print(
                "This mismatch is critical for Annoy. Please ensure LOCAL_MODEL_PATH points to the same type of model that built your Annoy index.",
                flush=True)
            # You might want to exit here if this is a severe mismatch
            # return False

    except Exception as e:
        print(f"Failed to load Sentence-BERT model from '{LOCAL_MODEL_PATH}': {e}", flush=True)
        print("Please ensure the model path is correct and it's a valid Sentence-Transformer model.", flush=True)
        return False

    print("Loading unique master customer names and embeddings map from CSV...", flush=True)
    if not os.path.exists(EMBEDDINGS_CSV_PATH):
        print(
            f"Error: Master Embeddings CSV not found at {EMBEDDINGS_CSV_PATH}. Ensure your previous script generated it.",
            flush=True)
        return False
    try:
        df_embeddings = pd.read_csv(EMBEDDINGS_CSV_PATH)
        master_unique_customer_names = df_embeddings[MASTER_COMPANY_NAME_COLUMN].tolist()

        # We don't need to load all embedding vectors into unique_name_to_embedding here,
        # as Annoy handles the embeddings itself. We just need the names for lookup.
        # This was part of the original Annoy setup, but not directly used for query here.
        # The key is that master_unique_customer_names list matches the Annoy index IDs.

        print(f"Loaded {len(master_unique_customer_names)} unique master names.", flush=True)
    except Exception as e:
        print(f"Failed to load unique master names from CSV: {e}", flush=True)
        return False

    print("Loading Annoy index...", flush=True)
    try:
        annoy_index = AnnoyIndex(EMBEDDING_DIM, ANNOY_METRIC)
        if annoy_index is None:
            print("ERROR: AnnoyIndex constructor returned None unexpectedly.", flush=True)
            return False
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize AnnoyIndex object: {e}", flush=True)
        return False

    if not os.path.exists(ANNOY_INDEX_FILE_PATH):
        print(
            f"Error: Annoy index file not found at {ANNOY_INDEX_FILE_PATH}. Ensure your previous script generated it.",
            flush=True)
        return False
    try:
        annoy_index.load(ANNOY_INDEX_FILE_PATH)
        print("Annoy index loaded.", flush=True)
    except Exception as e:
        print(f"Failed to load Annoy index from '{ANNOY_INDEX_FILE_PATH}': {e}", flush=True)
        if annoy_index is None:
            print("Reason for load failure: annoy_index object was None.", flush=True)
        print(
            "This often means the EMBEDDING_DIM in the script doesn't match the dimension the Annoy index was saved with.",
            flush=True)
        return False

    print("Loading master customer codes for lookup...", flush=True)
    try:
        df_master = pd.read_excel(MASTER_EXCEL_FILE)
        # Only load necessary columns to save memory
        df_master = df_master[[MASTER_COMPANY_NAME_COLUMN, MASTER_CUSTOMER_CODE_COLUMN]]
        master_name_to_code = pd.Series(df_master[MASTER_CUSTOMER_CODE_COLUMN].values,
                                        index=df_master[MASTER_COMPANY_NAME_COLUMN]).to_dict()
        print(f"Loaded {len(master_name_to_code)} master name-to-code mappings.", flush=True)
    except FileNotFoundError:
        print(f"Error: Master Excel file not found at {MASTER_EXCEL_FILE}. Please check the path.", flush=True)
        return False
    except Exception as e:
        print(f"Error loading master Excel file for name-to-code mapping: {e}", flush=True)
        return False

    return True


async def process_navision_company_async(session, semaphore, original_df_index, navision_company_name):
    """
    Asynchronously processes a single Navision company to find duplicates in the master list using Annoy,
    FuzzyWuzzy (for initial quick checks), and Gemini API for confirmation.
    """
    result = {
        'original_df_index': original_df_index,
        'is_duplicate': False,
        'matched_master_company_name': np.nan,
        'matched_master_customer_code': np.nan,
        'llm_explanation': np.nan,
        'flagged_for_review': False
    }

    if not navision_company_name:
        return result

    if should_exit:  # Check exit flag early
        raise asyncio.CancelledError("Graceful exit requested.")

    # Get embedding for the Navision company name
    try:
        navision_embedding = sentence_bert_model.encode(navision_company_name, convert_to_numpy=True)
    except Exception as e:
        result['llm_explanation'] = f"Embedding generation failed: {e}"
        result['flagged_for_review'] = True
        print(
            f"Warning: Failed to generate embedding for Navision company '{navision_company_name}': {e}. Skipping Annoy/LLM.",
            flush=True)
        return result

    if navision_embedding.shape[0] != EMBEDDING_DIM:
        result['llm_explanation'] = f"Embedding dimension mismatch: {navision_embedding.shape[0]} vs {EMBEDDING_DIM}"
        result['flagged_for_review'] = True
        print(
            f"Error: Navision embedding dimension ({navision_embedding.shape[0]}) generated by '{LOCAL_MODEL_PATH}' does not match configured Annoy index dimension ({EMBEDDING_DIM}). Skipping '{navision_company_name}'.",
            flush=True)
        return result

    # Find nearest neighbors in the master list using Annoy
    neighbor_ids, distances = annoy_index.get_nns_by_vector(
        navision_embedding, n=10, include_distances=True
    )

    best_llm_match = None
    best_llm_decision = False

    for neighbor_id, dist in zip(neighbor_ids, distances):
        if should_exit:
            raise asyncio.CancelledError("Graceful exit requested.")

        master_candidate_name = master_unique_customer_names[neighbor_id]
        cos_similarity = 1 - (dist ** 2 / 2)

        if cos_similarity < ANNOY_CANDIDATE_SIMILARITY_THRESHOLD:
            continue

        fuzzy_score = fuzz.ratio(navision_company_name.lower(), master_candidate_name.lower())
        if fuzzy_score >= FUZZY_MATCH_THRESHOLD_INITIAL:
            matched_code = master_name_to_code.get(master_candidate_name)
            result['is_duplicate'] = True
            result['matched_master_company_name'] = master_candidate_name
            result['matched_master_customer_code'] = matched_code
            result['llm_explanation'] = f"High fuzzy match ({fuzzy_score}) bypasses LLM."
            result['flagged_for_review'] = False
            print(
                f"  (Fuzzy) '{navision_company_name}' matched '{master_candidate_name}' (score {fuzzy_score}). Bypassing LLM.",
                flush=True)
            return result

        async with semaphore:
            is_same_llm, explanation_llm = await ask_gemini_if_same_company_async(
                session, navision_company_name, master_candidate_name
            )

        if is_same_llm:
            matched_code = master_name_to_code.get(master_candidate_name)
            best_llm_match = (master_candidate_name, matched_code, explanation_llm)
            best_llm_decision = True
            break

    if best_llm_decision:
        result['is_duplicate'] = True
        result['matched_master_company_name'] = best_llm_match[0]
        result['matched_master_customer_code'] = best_llm_match[1]
        result['llm_explanation'] = best_llm_match[2]
        result['flagged_for_review'] = (
                    analyze_for_flagging(navision_company_name, best_llm_match[0], best_llm_match[2],
                                         best_llm_decision) != "")
        print(
            f"  (LLM) '{navision_company_name}' confirmed duplicate of '{best_llm_match[0]}'. Flagged: {result['flagged_for_review']}",
            flush=True)
    else:
        result['is_duplicate'] = False
        result['llm_explanation'] = "No strong match found via Annoy/Fuzzy/LLM."
        if neighbor_ids:
            result['flagged_for_review'] = (
                        analyze_for_flagging(navision_company_name, "", result['llm_explanation'], False) != "")
        else:
            result['llm_explanation'] = "No relevant Annoy candidates found."
            result['flagged_for_review'] = True

        print(f"  '{navision_company_name}' not identified as duplicate. Flagged: {result['flagged_for_review']}",
              flush=True)

    return result


async def find_duplicates_in_navision_async():
    global dedup_llm_call_count, dedup_llm_cache_hits, should_exit

    signal.signal(signal.SIGINT, signal_handler)

    if not load_all_models_and_data_for_deduplication():
        print("Failed to load necessary components for deduplication. Exiting.", flush=True)
        return

    print("\n--- Starting Navision Deduplication Process ---", flush=True)

    try:
        df_navision = pd.read_excel(NAVISION_EXCEL_FILE)
        if NAVISION_COMPANY_NAME_COLUMN not in df_navision.columns:
            print(f"Error: Required column '{NAVISION_COMPANY_NAME_COLUMN}' not found in '{NAVISION_EXCEL_FILE}'.",
                  flush=True)
            print("Please ensure your Navision Excel file has this exact column header.", flush=True)
            return
    except FileNotFoundError:
        print(f"Error: Navision Excel file not found at {NAVISION_EXCEL_FILE}.", flush=True)
        return
    except Exception as e:
        print(f"Error loading Navision Excel file: {e}", flush=True)
        return

    df_navision[IS_DUPLICATE_COLUMN] = False
    df_navision[MATCHED_MASTER_COMPANY_NAME_COLUMN] = np.nan
    df_navision[MATCHED_MASTER_CUSTOMER_CODE_COLUMN] = np.nan
    df_navision[LLM_EXPLANATION_COLUMN] = np.nan
    df_navision[FLAGGED_FOR_REVIEW_COLUMN] = False

    unprocessed_indices_full = df_navision.index.tolist()
    processed_navision_count_from_checkpoint = load_dedup_checkpoint()

    if processed_navision_count_from_checkpoint >= len(unprocessed_indices_full):
        print("All Navision items appear to have been processed according to checkpoint. Proceeding to final save.",
              flush=True)
        indices_to_process_this_run = []
    else:
        # TEST_RUN_LIMIT = 100 # Uncomment this line for a test run of the first 100 unprocessed companies
        TEST_RUN_LIMIT = float('inf')  # Uncomment this line for a full run

        start_index_for_this_run = processed_navision_count_from_checkpoint
        end_index_for_this_run = min(len(unprocessed_indices_full), start_index_for_this_run + TEST_RUN_LIMIT)

        indices_to_process_this_run = unprocessed_indices_full[start_index_for_this_run:end_index_for_this_run]

        print(
            f"Resuming processing from Navision company {start_index_for_this_run + 1}. Processing {len(indices_to_process_this_run)} companies in this run.",
            flush=True)

    processed_count_this_session = 0
    start_time = time.time()

    tasks = []
    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(CONCURRENT_API_CALLS_LIMIT)

        for original_df_index in indices_to_process_this_run:
            if should_exit:  # Check should_exit before creating each task
                print("Graceful shutdown detected. Stopping task creation.", flush=True)
                break

            navision_company_name = str(df_navision.loc[original_df_index, NAVISION_COMPANY_NAME_COLUMN]).strip()
            tasks.append(process_navision_company_async(session, semaphore, original_df_index, navision_company_name))

        print(f"Initiating {len(tasks)} asynchronous tasks for Navision deduplication...", flush=True)

        try:
            # Gather tasks, handling cancellation
            # If should_exit was set, only await the tasks that were already created before the break
            async_results = await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            print("\nGraceful shutdown acknowledged. Finishing current operations and saving checkpoint...", flush=True)
            async_results = []  # No new results to process if cancelled early

        print("Applying results from asynchronous tasks to DataFrame...", flush=True)
        for result in async_results:
            if should_exit:  # Check should_exit before processing each result
                print("Graceful exit requested during result application. Breaking.", flush=True)
                break

            if isinstance(result, Exception):
                if isinstance(result, asyncio.CancelledError):
                    print(f"  Task cancelled: {result}", flush=True)
                    continue
                print(f"  Error in async task: {result}", flush=True)
                continue

            original_df_index = result['original_df_index']

            # Apply results to DataFrame
            df_navision.loc[original_df_index, IS_DUPLICATE_COLUMN] = result['is_duplicate']
            df_navision.loc[original_df_index, MATCHED_MASTER_COMPANY_NAME_COLUMN] = result[
                'matched_master_company_name']
            df_navision.loc[original_df_index, MATCHED_MASTER_CUSTOMER_CODE_COLUMN] = result[
                'matched_master_customer_code']
            df_navision.loc[original_df_index, LLM_EXPLANATION_COLUMN] = result['llm_explanation']
            df_navision.loc[original_df_index, FLAGGED_FOR_REVIEW_COLUMN] = result['flagged_for_review']

            processed_count_this_session += 1
            current_total_processed = processed_navision_count_from_checkpoint + processed_count_this_session

            if not should_exit and current_total_processed % CHECKPOINT_INTERVAL_COMPANIES == 0:
                save_dedup_checkpoint(current_total_processed)
            elif should_exit:  # If graceful exit requested, save checkpoint for any completed batch
                save_dedup_checkpoint(current_total_processed)
                # No need to break here, as the outer loop (creating tasks) already stopped.
                # We want to process all results that were already completed or in progress.

    final_processed_count = processed_navision_count_from_checkpoint + processed_count_this_session
    save_dedup_checkpoint(final_processed_count)  # Final save is always triggered

    print(
        f"\nNavision Deduplication Completed. Processed {processed_count_this_session} companies this session (total {final_processed_count}).",
        flush=True)
    print(f"Total LLM API calls made for deduplication: {dedup_llm_call_count}", flush=True)
    print(f"Total LLM cache hits for deduplication: {dedup_llm_cache_hits}", flush=True)
    print(f"Saving results to '{OUTPUT_DEDUP_EXCEL_FILE}'...", flush=True)

    MAX_EXCEL_SAVE_RETRIES = 5
    EXCEL_RETRY_DELAY_SECONDS = 10

    for attempt in range(1, MAX_EXCEL_SAVE_RETRIES + 1):
        try:
            df_navision.to_excel(OUTPUT_DEDUP_EXCEL_FILE, index=False)
            print(f"Deduplication results saved to '{OUTPUT_DEDUP_EXCEL_FILE}'.", flush=True)
            break
        except PermissionError as e:
            if attempt < MAX_EXCEL_SAVE_RETRIES:
                print(
                    f"Permission Error: '{OUTPUT_DEDUP_EXCEL_FILE}' is open. Please close it. Retrying in {EXCEL_RETRY_DELAY_SECONDS} seconds... (Attempt {attempt}/{MAX_EXCEL_SAVE_RETRIES})",
                    flush=True)
                time.sleep(EXCEL_RETRY_DELAY_SECONDS)
            else:
                print(
                    f"Final attempt failed: Permission Error: '{OUTPUT_DEDUP_EXCEL_FILE}' is still open after {MAX_EXCEL_SAVE_RETRIES} attempts. Please close it and manually save the DataFrame.",
                    flush=True)
        except Exception as e:
            print(f"An unexpected error occurred while saving to Excel: {e}", flush=True)
            break


# --- Main execution block ---
if __name__ == "__main__":
    try:
        from fuzzywuzzy import fuzz
    except ImportError:
        print("Error: 'fuzzywuzzy' and 'python-Levenshtein' are required. Please install them using:", flush=True)
        print("pip install fuzzywuzzy python-Levenshtein", flush=True)
        exit()

    # No multiprocessing.freeze_support() needed for single-process

    asyncio.run(find_duplicates_in_navision_async())