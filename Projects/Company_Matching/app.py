# app.py
from flask import Flask, render_template, request
import query_service # Import your query service module

app = Flask(__name__)

# --- Initialize the F.R.A.N.K. engine once when the Flask app starts ---
print("Initializing the F.R.A.N.K. engine...")
model_loaded = query_service.load_sentence_bert_model_global()
names_loaded = query_service.load_original_names_map_global(query_service.EMBEDDINGS_CSV_PATH)
annoy_loaded = query_service.load_annoy_index_global()

if not (model_loaded and names_loaded and annoy_loaded):
    print("FATAL ERROR: Failed to load all necessary components for F.R.A.N.K. engine.")
    print("Please check paths and file existence. Exiting.")
    # In a production environment, you might want to raise an exception or log heavily
    exit() # Exit the application if critical components fail to load

print("F.R.A.N.K. engine ready. Starting Flask application.")

@app.route('/', methods=['GET', 'POST'])
def index():
    query_name = ""
    results = []
    n_results = 5  # Default value for the slider
    min_similarity = 0.85 # Default value for the slider

    if request.method == 'POST':
        query_name = request.form.get('query_name', '').strip()
        try:
            n_results = int(request.form.get('n_results', n_results)) # Use default if not provided
            min_similarity = float(request.form.get('min_similarity', min_similarity)) # Use default if not provided

            # Enforce minimums for sliders (as set in HTML min attributes)
            if min_similarity < 0.5:
                min_similarity = 0.5
            if n_results < 1:
                n_results = 1

        except ValueError:
            # This handles cases where user might somehow input non-numeric values
            print("Invalid input for n_results or min_similarity. Using defaults.")
            # Defaults are already set, so no explicit assignment needed here unless you want to reset them

        if query_name:
            results = query_service.query_customer_names(query_name, n_results, min_similarity)
        else:
            results = [] # Clear results if query is empty or just whitespace

    return render_template('index.html',
                           query_name=query_name,
                           results=results,
                           n_results=n_results, # Pass current slider value back to template
                           min_similarity=min_similarity) # Pass current slider value back to template

if __name__ == '__main__':
    # Run in debug mode for development (auto-reloads on code changes, shows detailed errors)
    # Set debug=False for production deployments
    app.run(debug=True)