import pandas as pd
import numpy as np
import os # Import the os module for path manipulation

def clean_customer_rollup(file_path, sheet_name=0):
    """
    Cleans the 'CustomerRollup' column in an Excel sheet based on a set of rules.

    Args:
        file_path (str): The path to the Excel file.
        sheet_name (str or int, optional): The name or index of the sheet to read.
                                            Defaults to 0 (first sheet).

    Returns:
        pandas.DataFrame: The DataFrame with the 'CustomerRollup' column cleaned.
                          Returns None if the file cannot be read or required columns are missing.
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None

    required_columns = ['CustomerName', 'CustomerCode', 'CustomerParent', 'LTD Revenue', 'CustomerRollup']
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        print(f"Error: Missing required columns in the Excel file: {missing_cols}")
        return None

    # Make a copy to avoid SettingWithCopyWarning
    df_cleaned = df.copy()

    # --- Data Type Preprocessing ---
    # Ensure CustomerCode and CustomerParent are strings to handle mixed types (numbers, text)
    df_cleaned['CustomerCode'] = df_cleaned['CustomerCode'].astype(str).str.strip()
    df_cleaned['CustomerParent'] = df_cleaned['CustomerParent'].astype(str).str.strip()
    df_cleaned['CustomerRollup'] = df_cleaned['CustomerRollup'].astype(str).str.strip()


    # Convert 'LTD Revenue' to numeric, coercing errors to NaN. Fill NaN with 0 for comparisons.
    # We will use this 'temp_ltd_revenue' for comparison, but the original column might retain NaNs if desired elsewhere.
    df_cleaned['temp_ltd_revenue'] = pd.to_numeric(df_cleaned['LTD Revenue'], errors='coerce').fillna(0)

    # Replace 'nan' string representation from previous .astype(str) where values were truly NaN
    df_cleaned.replace('nan', np.nan, inplace=True)
    df_cleaned['CustomerParent'].replace('', np.nan, inplace=True) # Treat empty strings as NaN for parent lookup
    df_cleaned['CustomerRollup'].replace('', np.nan, inplace=True) # Treat empty strings as NaN for rollup group lookup


    # Get a list of unique initial CustomerRollup values to iterate through groups
    # We'll work on a copy of the original CustomerRollup values for grouping,
    # as we'll modify the column during iteration.
    initial_rollup_groups = df_cleaned['CustomerRollup'].dropna().unique()

    print(f"Processing {len(initial_rollup_groups)} unique CustomerRollup groups...")

    for group_id in initial_rollup_groups:
        # Select rows belonging to the current group
        group_rows_mask = (df_cleaned['CustomerRollup'] == group_id)
        group_df = df_cleaned[group_rows_mask].copy() # Work with a copy of the group

        # Get unique non-null CustomerParents within this specific group
        unique_parents_in_group = group_df['CustomerParent'].dropna().unique()
        num_unique_parents = len(unique_parents_in_group)

        # Initialize the chosen rollup for the group
        chosen_rollup_code = None

        # --- Apply Rules ---

        # Rule 1: If a group contains only one unique non-blank CustomerParent
        if num_unique_parents == 1:
            chosen_rollup_code = unique_parents_in_group[0]
            # print(f"Group '{group_id}': Rule 1 applied. Rollup set to '{chosen_rollup_code}'")

        # Rule 2: If a group is of size 1 (and Rule 1 didn't apply)
        elif len(group_df) == 1:
            chosen_rollup_code = group_df['CustomerCode'].iloc[0]
            # print(f"Group '{group_id}': Rule 2 applied. Rollup set to '{chosen_rollup_code}'")

        # Rule 4: If a group contains more than one unique CustomerParent value
        # This also implicitly handles cases where there are SOME parents, but not just one
        elif num_unique_parents > 1:
            # Filter for rows that *do* have a non-blank CustomerParent
            rows_with_parent = group_df[group_df['CustomerParent'].notna()]

            if not rows_with_parent.empty:
                # Find the row with the highest LTD Revenue among those with a parent
                # Then apply tie-breaker for CustomerCode alphabetically
                highest_revenue_parent_row = rows_with_parent.loc[
                    rows_with_parent.sort_values(by=['temp_ltd_revenue', 'CustomerCode'],
                                                 ascending=[False, True]).index[0]
                ]
                chosen_rollup_code = highest_revenue_parent_row['CustomerParent']
                # print(f"Group '{group_id}': Rule 4 applied. Rollup set to '{chosen_rollup_code}' (based on {highest_revenue_parent_row['CustomerCode']}'s parent)")
            else:
                # Fallback to Rule 3 if Rule 4 path resulted in no parents after all (unlikely given num_unique_parents > 1 check, but for robustness)
                # Find the CustomerCode associated with the highest LTD Revenue in the group
                # Apply tie-breaker for CustomerCode alphabetically
                highest_revenue_row = group_df.loc[
                    group_df.sort_values(by=['temp_ltd_revenue', 'CustomerCode'],
                                         ascending=[False, True]).index[0]
                ]
                chosen_rollup_code = highest_revenue_row['CustomerCode']
                # print(f"Group '{group_id}': Rule 3 (fallback in Rule 4) applied. Rollup set to '{chosen_rollup_code}'")


        # Rule 3: If a group does not have any values that point to a CustomerParent
        # This will be the fallback if none of the above rules apply (i.e., num_unique_parents is 0)
        else: # num_unique_parents == 0
            # Find the CustomerCode associated with the highest LTD Revenue in the group
            # Apply tie-breaker for CustomerCode alphabetically
            highest_revenue_row = group_df.loc[
                group_df.sort_values(by=['temp_ltd_revenue', 'CustomerCode'],
                                     ascending=[False, True]).index[0]
            ]
            chosen_rollup_code = highest_revenue_row['CustomerCode']
            # print(f"Group '{group_id}': Rule 3 applied. Rollup set to '{chosen_rollup_code}'")

        # Update the CustomerRollup for all rows in the original DataFrame for this group
        if chosen_rollup_code is not None:
            df_cleaned.loc[group_rows_mask, 'CustomerRollup'] = chosen_rollup_code
        else:
            print(f"Warning: No rollup code determined for group '{group_id}'. Leaving as is.")

    # Drop the temporary LTD Revenue column
    df_cleaned.drop(columns=['temp_ltd_revenue'], inplace=True)

    return df_cleaned

if __name__ == "__main__":
    # Original file path
    excel_file_path = r"C:\Users\JackLindgren\OneDrive - Burwell Material Handling\Desktop\Data\Completed\Reducedcustomerlistwrollup_final.xlsx"

    # Specify the sheet name if it's not the first one (index 0)
    sheet_name_to_process = 0 # Assuming data is on the first sheet

    print(f"Starting cleanup for file: {excel_file_path}")

    cleaned_df = clean_customer_rollup(excel_file_path, sheet_name=sheet_name_to_process)

    if cleaned_df is not None:
        print("\nCleanup complete! Here are the first few rows of the cleaned data:")
        print(cleaned_df.head())

        # Construct the new output file path in the same directory
        directory = os.path.dirname(excel_file_path)
        original_filename_without_ext = os.path.splitext(os.path.basename(excel_file_path))[0]
        output_file_name = f"{original_filename_without_ext}_cleaned_rollup.xlsx"
        output_file_path = os.path.join(directory, output_file_name)

        # Save the cleaned data to a new file in the same folder
        try:
            cleaned_df.to_excel(output_file_path, index=False, sheet_name='CleanedRollupData')
            print(f"\nCleaned data successfully saved to: {output_file_path}")
        except Exception as e:
            print(f"Error saving cleaned data: {e}")
    else:
        print("\nCleaning process could not be completed due to errors.")