import candatascience as cds

def arrange_ds():
    """"""
    # File path - assuming the text file is in the same directory as this script
    file_path = 'file_path'
    full_file_path= 'full_file_path'

    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
    
    pass


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# File path - assuming the text file is in the same directory as this script
file_path = 'file_path'
full_file_path= 'full_file_path'

def analyze_csv():
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    # Read the raw text file
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Extract the header line (feature names)
    header_line = lines[0].strip()
    if header_line.startswith("i have a dataset"):
        # Extract the actual header from the description text
        start_idx = header_line.find("_STATE")
        if start_idx != -1:
            header_line = header_line[start_idx:]

    # Parse the header to get column names
    column_names = header_line.split(',')

    # Find the data lines
    data_lines = []
    for line in lines:
        if line.strip().startswith("1.0,") or line.strip().startswith("A2 is this:"):
            # Extract the actual data if it's prefixed with description
            if "A2 is this:" in line:
                data_start = line.find("A2 is this:")
                line = line[data_start + len("A2 is this:"):].strip()
            data_lines.append(line.strip())

    # Create a proper DataFrame
    data_values = [row.split(',') for row in data_lines]
    df = pd.DataFrame(data_values, columns=column_names)

    # Convert to appropriate data types
    for col in df.columns:
        # Try to convert to numeric, if fails keep as is
        df[col] = pd.to_numeric(df[col], errors='ignore')

    # Replace 'nan' strings with actual NaN values
    df.replace('nan', np.nan, inplace=True)

    # Display basic information
    print("\n===== Basic Dataset Information =====")
    print(f"Dataset shape: {df.shape}")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")

    # Count NaN values for each column
    print("\n===== NaN Values Analysis =====")
    nan_counts = df.isna().sum().sort_values(ascending=False)
    nan_percentage = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)

    # Create a DataFrame to display NaN counts and percentages
    nan_summary = pd.DataFrame({
        'NaN Count': nan_counts,
        'NaN Percentage': nan_percentage.round(2)
    })

    # Display columns with NaN values (filtering out columns with 0 NaNs)
    nan_columns = nan_summary[nan_summary['NaN Count'] > 0]
    print(f"Number of columns with missing values: {len(nan_columns)}")
    print("\nColumns with most missing values (top 20):")
    print(nan_columns.head(20))

    # Save NaN analysis to CSV
    nan_columns.to_csv('nan_analysis.csv')
    print("NaN analysis saved to 'nan_analysis.csv'")

    # Basic statistics for numeric columns
    print("\n===== Numeric Data Statistics =====")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        numeric_stats = df[numeric_cols].describe()
        print(numeric_stats)
        numeric_stats.to_csv('numeric_statistics.csv')
        print("Numeric statistics saved to 'numeric_statistics.csv'")
    else:
        print("No numeric columns found for statistics.")

    # Visualizations
    print("\n===== Creating Visualizations =====")

    # Top 20 columns with most NaN values
    plt.figure(figsize=(12, 8))
    top_nan_columns = nan_columns.head(20)
    sns.barplot(x=top_nan_columns['NaN Percentage'], y=top_nan_columns.index)
    plt.title('Top 20 Columns with Most Missing Values')
    plt.xlabel('Missing Values (%)')
    plt.tight_layout()
    plt.savefig('missing_values_chart.png')
    print("Missing values chart saved to 'missing_values_chart.png'")

    # Save cleaned DataFrame to a proper CSV
    df.to_csv('cleaned_data.csv', index=False)
    print("Cleaned data saved to 'cleaned_data.csv'")

    return df


if __name__ == "__main__":
    df = analyze_csv()
    print("\nAnalysis complete! Results saved to CSV files and PNG images.")


