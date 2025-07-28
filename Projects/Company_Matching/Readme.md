**Purpose of each file:**

**Bert-only.py:** used to take customer names and turn them into an embeddings csv with 768 dimensions. This file also attempts a rudementarary embedding similarity merging system,
  however this was found to be inferior to using the annoy to LLM method, only used the csv from this code.

**Annoy.py:** uses the embeddings from Bert-Only to create an ANNOY model for rapid similarity checks

**Parentcodetorollupcode:** Takes a excel file containing a column for ParentCode and CustomerCode and creates a new column RollupCode
  that will group companies based on their Parent-Child Relationships that have been pre-established

**Query_Service.py:** Code used to manually interact with the ANNOY.py model

**app.py:** Loads the Query_Service.py model and starts a locally hosted webpage based on the index.html file in Templates

**templates/Index.html:** Has the HTML for the F.R.A.N.K. engine, used by app.py

**Deduplication_companies_Rollup.py:** Utilizes the files from the bert and annoy models. The annoy outputs are then set 
  to make comparison of each result with the query name by an LLM (currently gemeni flash 1.5). The LLM outputs are
  then scraped for a yes or no and the explanation. Code is set to save progress and cache results. This cache
  is also used to insure that duplicate comparisons are not made and save on API calls. (Caches automatically
  every 100 names processed).

**Post_Deduplication_Cleanup.py:** Takes the output from deduplication process (including some manual checks occurin after its use) 
  and insures that the correct CustomerCode is used as CustomerRollup with the following algorithm:
  
  1. Determine groups where the CustomerRollup Value is the same
  2. if group size is 1, set rollup as self
  3. if Group contains only one Parentcode (only one parent noted in parent child relationship), set group to that ParentCode
  4. If Group contains 2+ ParentCodes, filter by CustomerCodes with a non-null ParentCode, Use the ParentCode of the CustomerCode with highest LTD Revenue
  5. If group Contains NO ParentCodes (no Parent-Child relationships), Set CustomerRollup to CustomerCode with the Highest LTD Revenue
  6. Any ties in above system are broken using earliest in alphabetical order

**Deduplicate_Navision.py:** Takes the names from one excel file and compares them to the Main customer list (using annoy and embeddings from Bert-only and .ann from Annoy.py.
any very close fuzzy matches will be marked as confident matches (over 0.95). Otherwise, matches will be made by an LLM. If a match is found at any point, the code moves to the next customername.
Creates cahing system for llm calls similar to Deduplication_companies_Rollup.py. Producess flags for ifsame, humanreview, and provides name and code of matches.

