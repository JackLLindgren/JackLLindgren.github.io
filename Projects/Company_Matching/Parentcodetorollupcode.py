import pandas as pd

# Define the file path
file_path = r"Insert File Path"

# Read the Excel file into a pandas DataFrame
df = pd.read_excel(file_path)

# Initialize the 'CustomerRollup' column
df['CustomerRollup'] = ''

# Fill 'CustomerRollup' based on 'CustomerParent'
# If 'CustomerParent' exists, copy its value
df['CustomerRollup'] = df['CustomerParent'].fillna(df['CustomerRollup'])

# Get all unique CustomerParent codes that actually exist in the 'CustomerParent' column
unique_parent_codes = df['CustomerParent'].dropna().unique()

# Iterate through the DataFrame to apply the logic for parent companies
for index, row in df.iterrows():
    # If a company's CustomerCode is present in the list of unique_parent_codes
    # and it doesn't already have a CustomerParent assigned (meaning it's a top-level parent itself)
    if row['CustomerCode'] in unique_parent_codes and pd.isna(row['CustomerParent']):
        # Then its CustomerRollup should be its own CustomerCode
        df.loc[index, 'CustomerRollup'] = row['CustomerCode']

# Save the updated DataFrame to a new Excel file
output_file_path = r"C:\Users\JackLindgren\Downloads\Reducedcustomerlistwrollupinitialized.xlsx"
df.to_excel(output_file_path, index=False)

print(f"CustomerRollup column created and saved to: {output_file_path}")