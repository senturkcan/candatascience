import candatascience as cds
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

import os
import pandas as pd
from pathlib import Path

def extract(label_column, dataset_path="", dataset_full_path="", file_type=None):
    """
    Extract features and labels from various dataset file formats.
    
    Parameters:
    -----------
    label_column : int or str
        Column index (int) or column name (str) for the target/label variable
    dataset_path : str, optional
        Relative path to the dataset file
    dataset_full_path : str, optional
        Absolute path to the dataset file
    file_type : str, optional
        File format: 'csv', 'xlsx', 'xls', 'json', 'parquet', 'feather', 'pickle', 'tsv'
        If None, file type is auto-detected from file extension
    
    Returns:
    --------
    tuple : (ds, x, y)
        ds: Full DataFrame
        x: Feature DataFrame (without label column)
        y: Label Series
    """
    
    # Determine which path to use
    file_path = dataset_full_path if dataset_full_path else dataset_path
    
    # File path validation
    if not file_path:
        raise ValueError("Error: Provide either dataset_path or dataset_full_path")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: File '{file_path}' not found")
    
    # Auto-detect file type from extension if not provided
    if file_type is None:
        # Get extension from the rightmost dot (handles files like "data.backup.csv")
        file_extension = os.path.splitext(file_path)[1].lower()
        
        # Map extensions to file types
        extension_map = {
            '.csv': 'csv',
            '.tsv': 'tsv',
            '.txt': 'csv',  # Assume txt is comma-delimited by default
            '.xlsx': 'xlsx',
            '.xls': 'xls',
            '.json': 'json',
            '.parquet': 'parquet',
            '.pq': 'parquet',
            '.feather': 'feather',
            '.pkl': 'pickle',
            '.pickle': 'pickle',
            '.h5': 'hdf',
            '.hdf': 'hdf',
            '.hdf5': 'hdf',
            '.dta': 'stata',
            '.sas7bdat': 'sas',
            '.sav': 'spss'
        }
        
        file_type = extension_map.get(file_extension)
        
        if file_type is None:
            raise ValueError(f"Cannot auto-detect file type from extension '{file_extension}'. "
                           f"Please specify file_type parameter. "
                           f"Supported extensions: {', '.join(extension_map.keys())}")
    else:
        file_type = file_type.lower()
    
    # Read file based on format with timeout consideration
    print(f"Loading dataset from '{file_path}' (format: {file_type})...")
    
    try:
        if file_type == "csv":
            ds = pd.read_csv(file_path, delimiter=",", low_memory=False)
        
        elif file_type == "tsv":
            ds = pd.read_csv(file_path, delimiter="\t", low_memory=False)
        
        elif file_type in ["xlsx", "xls"]:
            ds = pd.read_excel(file_path, engine='openpyxl' if file_type == 'xlsx' else None)
        
        elif file_type == "json":
            ds = pd.read_json(file_path)
        
        elif file_type == "parquet":
            ds = pd.read_parquet(file_path, engine='pyarrow')
        
        elif file_type == "feather":
            ds = pd.read_feather(file_path)
        
        elif file_type in ["pickle", "pkl"]:
            ds = pd.read_pickle(file_path)
        
        elif file_type in ["hdf", "h5", "hdf5"]:
            ds = pd.read_hdf(file_path)
        
        elif file_type in ["stata", "dta"]:
            ds = pd.read_stata(file_path)
        
        elif file_type == "sas":
            ds = pd.read_sas(file_path)
        
        elif file_type in ["spss", "sav"]:
            ds = pd.read_spss(file_path)
        
        else:
            raise ValueError(f"Unsupported file type: '{file_type}'. "
                           f"Supported formats: csv, tsv, xlsx, xls, json, parquet, "
                           f"feather, pickle, hdf, stata, sas, spss")
        
        print(f"Dataset loaded successfully: {ds.shape[0]} rows, {ds.shape[1]} columns")
    
    except MemoryError:
        raise RuntimeError(f"Error: Dataset is too large to load into memory. "
                         f"File: '{file_path}'. "
                         f"Consider using chunking, sampling, or a more memory-efficient format like Parquet. "
                         f"For very large datasets, consider using Dask or processing in chunks.")
    
    except pd.errors.EmptyDataError:
        raise RuntimeError(f"Error: File '{file_path}' is empty or contains no data.")
    
    except pd.errors.ParserError as e:
        raise RuntimeError(f"Error parsing file '{file_path}': {str(e)}. "
                         f"The file may be corrupted, too large, or the format doesn't match the specified file_type. "
                         f"For large files, consider increasing system resources or using a different format.")
    
    except (IOError, OSError) as e:
        raise RuntimeError(f"Error reading file '{file_path}': {str(e)}. "
                         f"This may occur if the file is too large, access is denied, or the disk is full.")
    
    except Exception as e:
        error_msg = str(e).lower()
        if 'timeout' in error_msg or 'time out' in error_msg or 'too long' in error_msg:
            raise RuntimeError(f"Error: Operation timed out while reading '{file_path}'. "
                             f"The dataset may be too large. Consider using a smaller subset, "
                             f"chunking the data, or using a more efficient format like Parquet.")
        else:
            raise RuntimeError(f"Error reading file '{file_path}': {str(e)}")
    
    # Extract features and labels
    if isinstance(label_column, int):
        # Validate column index
        if label_column < 0 or label_column >= len(ds.columns):
            raise IndexError(f"Column index {label_column} is out of range. "
                           f"Dataset has {len(ds.columns)} columns (indices 0-{len(ds.columns)-1})")
        
        # Select label by column index
        y = ds.iloc[:, label_column]
        x = ds.drop(ds.columns[label_column], axis=1)
    
    elif isinstance(label_column, str):
        # Select label by column name
        if label_column not in ds.columns:
            raise KeyError(f"Column '{label_column}' not found in dataset. "
                         f"Available columns: {list(ds.columns)}")
        y = ds[label_column]
        x = ds.drop(label_column, axis=1)
    
    else:
        raise TypeError("label_column must be int (column index) or str (column name)")
    
    return ds, x, y



def analyze_csv():
    # Check if file exists
    if not os.path.exists(dataset_path):
        print(f"Error: File '{dataset_path}' not found.")
        return

    # Read the raw text file
    with open(dataset_path, 'r') as f:
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






