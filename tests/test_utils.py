import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os




import os
import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path

def extract(label_column = None, dataset_path="", dataset_full_path="", file_type=None, 
            header='infer', clean_data=True, remove_duplicate=False, sql_query=None, sql_table=None):
    """
    Extract features and labels from various dataset file formats with automatic cleaning.
    
    Parameters:
    -----------
    label_column : int or str
        Column index (int) or column name (str) for the target/label variable
    dataset_path : str, optional
        Relative path to the dataset file
    dataset_full_path : str, optional
        Absolute path to the dataset file
    file_type : str, optional
        File format: 'csv', 'xlsx', 'xls', 'json', 'parquet', 'feather', 'pickle', 'tsv', 'sql'
        If None, file type is auto-detected from file extension
    header : int, list of int, 'infer', or None, default 'infer'
        Row number(s) to use as column names. 'infer' auto-detects, None means no header
    clean_data : bool, default True
        If True, applies automatic data cleaning (convert numerics, clean strings, handle missing)
    sql_query : str, optional
        SQL query to execute (for SQL databases). If None, reads entire table specified by sql_table
    sql_table : str, optional
        Table name to read from SQL database (required if sql_query is None)
    
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
    
    file_path = os.path.abspath(file_path)
    
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
            '.sav': 'spss',
            '.db': 'sql',
            '.sqlite': 'sql',
            '.sqlite3': 'sql',
            '.sql': 'sql'
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
            ds = pd.read_csv(file_path, delimiter=",", low_memory=False, header=header)
        
        elif file_type == "tsv":
            ds = pd.read_csv(file_path, delimiter="\t", low_memory=False, header=header)
        
        elif file_type in ["xlsx", "xls"]:
            ds = pd.read_excel(file_path, engine='openpyxl' if file_type == 'xlsx' else None, header=header)
        
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
        
        elif file_type == "sql":
            # Handle SQL databases
            conn = sqlite3.connect(file_path)
            try:
                if sql_query:
                    ds = pd.read_sql_query(sql_query, conn)
                elif sql_table:
                    ds = pd.read_sql_table(sql_table, conn)
                else:
                    # Try to get the first table if no query or table specified
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                    if tables:
                        first_table = tables[0][0]
                        print(f"No sql_query or sql_table specified. Reading first table: '{first_table}'")
                        ds = pd.read_sql_table(first_table, conn)
                    else:
                        raise ValueError("No tables found in SQL database")
            finally:
                conn.close()
        
        else:
            raise ValueError(f"Unsupported file type: '{file_type}'. "
                           f"Supported formats: csv, tsv, xlsx, xls, json, parquet, "
                           f"feather, pickle, hdf, stata, sas, spss, sql")
        
        print(f"Dataset loaded successfully.")
    
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
    
    # Clean data if requested
    if clean_data:
        print("Cleaning dataset...")
        if remove_duplicate:
            ds = _clean_dataframe(ds, remove_duplicate = True)
        else:
            ds = _clean_dataframe(ds, remove_duplicate = False)
        print("Initial data cleaning completed")
    
    # Extract features and labels

    if label_column == None:
        pass
    elif isinstance(label_column, int):
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


def _clean_dataframe(df, remove_duplicate):
    """
    Apply automatic data cleaning to a DataFrame.
    
    - Converts numeric-looking columns to numeric types
    - Cleans string columns (strip whitespace, lowercase)
    - Handles missing values appropriately
    - Removes duplicate rows
    """
    df_cleaned = df.copy()
    
    for col in df_cleaned.columns:
        # Skip if column is already numeric
        if pd.api.types.is_numeric_dtype(df_cleaned[col]):
            continue
        
        # Try to convert to numeric
        converted = pd.to_numeric(df_cleaned[col], errors='coerce')
        
        # If most values converted successfully, use numeric type
        non_null_count = converted.notna().sum()
        original_non_null = df_cleaned[col].notna().sum()
        
        if original_non_null > 0 and (non_null_count / original_non_null) > 0.5:
            # More than 50% of non-null values are numeric
            df_cleaned[col] = converted
        else:
            # Treat as string column
            if df_cleaned[col].dtype == 'object':
                # Clean string data: strip whitespace and convert to lowercase
                df_cleaned[col] = df_cleaned[col].apply(
                    lambda x: str(x).strip().lower() if pd.notna(x) and x != '' else x
                )
                
                # Replace empty strings with NaN
                df_cleaned[col] = df_cleaned[col].replace('', np.nan)
    if remove_duplicate:
        # Remove duplicate rows
        initial_rows = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates()
        removed_duplicates = initial_rows - len(df_cleaned)
        
        if removed_duplicates > 0:
            print(f"  - Removed {removed_duplicates} duplicate rows")
    
    return df_cleaned


# Example usage:
    # Auto-detect file type from extension
    # ds, x, y = extract(label_column="target", dataset_path="data.csv")
    
    # With custom header row
    # ds, x, y = extract(label_column="target", dataset_path="data.csv", header=0)
    
    # Without cleaning
    # ds, x, y = extract(label_column=0, dataset_path="data.xlsx", clean_data=False)
    
    # SQL database with query
    # ds, x, y = extract(label_column="label", dataset_path="database.db", 
    #                    sql_query="SELECT * FROM customers WHERE age > 18")
    
    # SQL database with table name
    # ds, x, y = extract(label_column="target", dataset_path="database.db", 
    #                    sql_table="training_data")
    
    pass





def initial_analysis(df):
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
    nan_summary = pd.df({
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

    pass





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

def main_analysis(df, label_column=None, task_type='regression', top_n_features=10):

    """
    Perform comprehensive Exploratory Data Analysis with feature relationship insights after missig value handling.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to analyze (assumes already cleaned if extract() was used with clean_data=True)
    label_column : str or int, optional
        Target/label column name or index. If None, only descriptive analysis is performed
    task_type : str, default 'auto'
        Type of ML task: 'classification', 'regression'
    top_n_features : int, default 10
        Number of top features to display in importance analysis
    
    Returns:
    --------
    dict : Analysis results containing insights for modeling decisions
    """
    

    
    print("="*70)
    print("EXPLORATORY DATA ANALYSIS REPORT")
    print("="*70)
    
    results = {}
    
    # 1. BASIC DATASET INFORMATION
    print("\n[1] DATASET OVERVIEW")
    print("-"*70)

    print(f"Dataset shape: {df.shape}")
    print(f"Features(x) shape: {x.shape}")
    print(f"Label(y) shape: {y.shape}")

    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    

    results['dataset_insights'] = {
        'rows': df.shape[0],
        'columns': df.shape[1],
        'memory_mb': df.memory_usage(deep=True).sum() / 1024**2
    }
    
    # 2. DATA TYPES
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"\nNumeric columns: {len(numeric_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")
    
    results['column_types'] = {
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols
    }
    
    # 3. MISSING DATA INSIGHTS
    print("\n[2] MISSING DATA STATUS")
    print("-"*70)
    missing = df.isnull().sum()
    total_missing = missing.sum()
    
    if total_missing > 0:
        missing_pct = (missing / len(df)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing,
            'Percentage': missing_pct
        }).sort_values('Percentage', ascending=False)
        missing_df = missing_df[missing_df['Missing_Count'] > 0]
        print(missing_df.to_string())
        
        print(f"\n⚠️  Total missing values: {total_missing} ({(total_missing / (df.shape[0] * df.shape[1]) * 100):.2f}% of dataset)")
    else:
        print("✓ No missing values detected")
    
    results['missing_data'] = missing_df.to_dict() if total_missing > 0 else {}
    
    # 4. TARGET VARIABLE ANALYSIS
    if label_column is not None:
        print("\n[3] TARGET VARIABLE ANALYSIS")
        print("-"*70)
        
        # Get target column
        if isinstance(label_column, int):
            target = df.iloc[:, label_column]
            target_name = df.columns[label_column]
        else:
            target = df[label_column]
            target_name = label_column
        
        print(f"Target: {target_name}")
        
        #  task type
        if label_column is not None:
            task_type = 'classification'
        print(f"Task Type: {task_type.upper()}")
        results['task_type'] = task_type
        
        if task_type == 'classification':
            print(f"\nClass Distribution:")
            class_dist = target.value_counts()
            class_dist_pct = target.value_counts(normalize=True)
            
            for cls, count in class_dist.items():
                print(f"  {cls}: {count} ({class_dist_pct[cls]*100:.1f}%)")
            
            # Check for class imbalance
            min_class_pct = class_dist_pct.min() * 100
            max_class_pct = class_dist_pct.max() * 100
            imbalance_ratio = max_class_pct / min_class_pct if min_class_pct > 0 else float('inf')
            
            print(f"\nClass Imbalance Ratio: {imbalance_ratio:.2f}:1")
            
            if imbalance_ratio > 3:
                print("⚠️  IMBALANCED DATASET detected!")
                print("💡 Recommendations:")
                print("   - Use stratified train-test split")
                print("   - Consider SMOTE, class weights, or resampling")
                print("   - Use metrics: F1-score, ROC-AUC, Precision-Recall (not just accuracy)")
                print("   - Models: XGBoost/LightGBM with scale_pos_weight")
                results['imbalance_detected'] = True
            else:
                print("✓ Classes are relatively balanced")
                print("💡 Can use standard train-test split and accuracy metrics")
                results['imbalance_detected'] = False
            
            results['class_distribution'] = class_dist.to_dict()
            
        elif task_type == 'regression': 
            print(f"\nTarget Statistics:")
            print(target.describe())
            
            # Check target distribution
            target_skew = target.skew()
            print(f"\nTarget Skewness: {target_skew:.3f}")
            
            if abs(target_skew) > 1:
                print("⚠️  Target is highly skewed!")
                print("💡 Recommendations:")
                print("   - Consider log transformation of target")
                print("   - Models: Tree-based (XGBoost, RandomForest) handle skewness better")
                print("   - For linear models: transform target then inverse transform predictions")
                results['target_skewed'] = True
            else:
                print("✓ Target distribution is acceptable")
                results['target_skewed'] = False
        
        # 5. FEATURE IMPORTANCE (using Mutual Information)
        print("\n[4] FEATURE IMPORTANCE ANALYSIS")
        print("-"*70)
        
        # Prepare features for mutual information
        X_analysis = df.drop(columns=[target_name])
        
        # Handle categorical variables with label encoding for MI calculation
        X_encoded = X_analysis.copy()
        for col in categorical_cols:
            if col in X_encoded.columns:
                X_encoded[col] = pd.factorize(X_encoded[col])[0]
        
        # Fill missing values temporarily for MI calculation
        X_encoded = X_encoded.fillna(X_encoded.median() if len(numeric_cols) > 0 else -1)
        
        # Calculate mutual information
        if task_type == 'classification':
            mi_scores = mutual_info_classif(X_encoded, target, random_state=42)
        else:
            mi_scores = mutual_info_regression(X_encoded, target, random_state=42)
        
        mi_scores = pd.Series(mi_scores, index=X_encoded.columns).sort_values(ascending=False)
        
        print(f"\nTop {top_n_features} Most Important Features (Mutual Information):")
        print(mi_scores.head(top_n_features))
        
        # Identify low importance features
        low_importance = mi_scores[mi_scores < 0.01]
        if len(low_importance) > 0:
            print(f"\n⚠️  {len(low_importance)} features have very low importance (<0.01)")
            print("💡 Consider feature selection to reduce dimensionality")
        
        results['feature_importance'] = mi_scores.to_dict()
        
        # 6. FEATURE CORRELATIONS
        print("\n[5] FEATURE CORRELATIONS WITH TARGET")
        print("-"*70)
        
        numeric_features = [col for col in numeric_cols if col != target_name]
        
        if len(numeric_features) > 0:
            correlations = df[numeric_features].corrwith(target).sort_values(ascending=False)
            correlations = correlations.dropna()
            
            print("\nTop 5 Positive Correlations:")
            print(correlations.head(5))
            
            print("\nTop 5 Negative Correlations:")
            print(correlations.tail(5))
            
            # Check for weak correlations
            weak_corr = correlations[abs(correlations) < 0.1]
            if len(weak_corr) > 0:
                print(f"\n💡 {len(weak_corr)} features have weak linear correlation with target")
                print("   Consider non-linear models (trees, neural networks)")
            
            results['correlations'] = correlations.to_dict()
        
        # 7. MULTICOLLINEARITY CHECK
        print("\n[6] MULTICOLLINEARITY ANALYSIS")
        print("-"*70)
        
        if len(numeric_features) > 1:
            corr_matrix = df[numeric_features].corr().abs()
            
            # Find highly correlated feature pairs
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.8:
                        high_corr_pairs.append({
                            'feature_1': corr_matrix.columns[i],
                            'feature_2': corr_matrix.columns[j],
                            'correlation': corr_matrix.iloc[i, j]
                        })
            
            if high_corr_pairs:
                print(f"⚠️  Found {len(high_corr_pairs)} highly correlated feature pairs (>0.8):")
                for pair in high_corr_pairs[:5]:
                    print(f"  {pair['feature_1']} ↔ {pair['feature_2']}: {pair['correlation']:.3f}")
                print("\n💡 Recommendations:")
                print("   - Remove one from each pair OR use PCA/feature selection")
                print("   - Tree-based models handle multicollinearity better than linear models")
                results['multicollinearity_detected'] = True
            else:
                print("✓ No significant multicollinearity detected")
                results['multicollinearity_detected'] = False
            
            results['high_correlations'] = high_corr_pairs
    
    # 8. DIMENSIONALITY INSIGHTS
    print("\n[7] DIMENSIONALITY ANALYSIS")
    print("-"*70)
    
    n_features = df.shape[1] - (1 if label_column is not None else 0)
    n_samples = df.shape[0]
    ratio = n_samples / n_features
    
    print(f"Samples-to-Features Ratio: {ratio:.1f}:1")
    

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    return results

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class OutlierDetector:
    """
    Detects anomalies/outliers in datasets using multiple methods.
    Saves flagged rows for review and provides easy removal functionality.
    """
    
    def __init__(self, df, output_file='potential_mistakes.csv'):
        """
        Initialize the detector.
        
        Args:
            df: pandas DataFrame to analyze
            output_file: CSV filename to save potential mistakes
        """
        self.df = df.copy()
        self.df['original_index'] = df.index
        self.output_file = output_file
        self.outliers = pd.DataFrame()
        
    def detect_outliers(self, methods=['zscore', 'iqr', 'isolation'], 
                       z_threshold=3, iqr_multiplier=1.5):
        """
        Detect outliers using multiple methods.
        
        Args:
            methods: List of methods to use ['zscore', 'iqr', 'isolation']
            z_threshold: Z-score threshold (default: 3)
            iqr_multiplier: IQR multiplier (default: 1.5)
        """
        outlier_indices = set()
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if 'original_index' in numeric_cols:
            numeric_cols.remove('original_index')
        
        if not numeric_cols:
            print("No numeric columns found for outlier detection!")
            return self
        
        results = []
        
        # Z-Score Method
        if 'zscore' in methods:
            print(f"Applying Z-Score method (threshold={z_threshold})...")
            for col in numeric_cols:
                z_scores = np.abs(stats.zscore(self.df[col].dropna()))
                outlier_mask = z_scores > z_threshold
                outlier_idx = self.df[col].dropna().index[outlier_mask]
                
                for idx in outlier_idx:
                    outlier_indices.add(idx)
                    results.append({
                        'original_index': self.df.loc[idx, 'original_index'],
                        'method': 'Z-Score',
                        'column': col,
                        'value': self.df.loc[idx, col],
                        'z_score': z_scores[self.df[col].dropna().index.get_loc(idx)],
                        'reason': f'Z-score={z_scores[self.df[col].dropna().index.get_loc(idx)]:.2f} > {z_threshold}'
                    })
        
        # IQR Method
        if 'iqr' in methods:
            print(f"Applying IQR method (multiplier={iqr_multiplier})...")
            for col in numeric_cols:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - iqr_multiplier * IQR
                upper_bound = Q3 + iqr_multiplier * IQR
                
                outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                outlier_idx = self.df[outlier_mask].index
                
                for idx in outlier_idx:
                    outlier_indices.add(idx)
                    val = self.df.loc[idx, col]
                    results.append({
                        'original_index': self.df.loc[idx, 'original_index'],
                        'method': 'IQR',
                        'column': col,
                        'value': val,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound,
                        'reason': f'Value {val:.2f} outside [{lower_bound:.2f}, {upper_bound:.2f}]'
                    })
        
        # Isolation Forest Method
        if 'isolation' in methods:
            try:
                from sklearn.ensemble import IsolationForest
                print("Applying Isolation Forest method...")
                
                iso_data = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                predictions = iso_forest.fit_predict(iso_data)
                
                outlier_mask = predictions == -1
                outlier_idx = self.df[outlier_mask].index
                
                for idx in outlier_idx:
                    outlier_indices.add(idx)
                    results.append({
                        'original_index': self.df.loc[idx, 'original_index'],
                        'method': 'Isolation Forest',
                        'column': 'Multiple',
                        'value': 'N/A',
                        'reason': 'Flagged as anomaly by Isolation Forest'
                    })
            except ImportError:
                print("Scikit-learn not available. Skipping Isolation Forest method.")
        
        # Create outliers DataFrame
        if results:
            self.outliers = pd.DataFrame(results)
            
            # Add all row data for flagged indices
            full_outliers = []
            for idx in sorted(outlier_indices):
                row_data = self.df.loc[idx].to_dict()
                flags = self.outliers[self.outliers['original_index'] == row_data['original_index']]
                row_data['detection_methods'] = ', '.join(flags['method'].unique())
                row_data['flagged_columns'] = ', '.join(flags['column'].unique())
                row_data['reasons'] = ' | '.join(flags['reason'].unique())
                full_outliers.append(row_data)
            
            self.outliers = pd.DataFrame(full_outliers)
            
            # Save to CSV
            self.outliers.to_csv(self.output_file, index=False)
            print(f"\n✓ Found {len(self.outliers)} potential mistakes")
            print(f"✓ Saved to '{self.output_file}'")
            print(f"\nTop flagged rows:")
            print(self.outliers[['original_index', 'detection_methods', 'flagged_columns']].head(10))
        else:
            print("No outliers detected!")
        
        return self
    
    def get_outliers(self):
        """Return the DataFrame of detected outliers."""
        return self.outliers
    
    @staticmethod
    def remove_rows(df, indices_to_remove):
        """
        Remove confirmed mistakes from the dataset after manual review.
        
        Args:
            df: DataFrame to clean
            indices_to_remove: List of row indices to remove
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.drop(indices_to_remove, errors='ignore')
        removed_count = len(df) - len(df_clean)
        print(f"✓ Removed {removed_count} rows")
        print(f"  Original size: {len(df)}, New size: {len(df_clean)}")
        return df_clean


# ============= USAGE EXAMPLE =============

if __name__ == "__main__":
    # Example 1: Load your data
    # df = pd.read_csv('your_data.csv')
    
    # Creating sample data with outliers for demonstration
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.normal(30, 5, 100).tolist() + [150, -10, 200],  # outliers
        'salary': np.random.normal(50000, 10000, 100).tolist() + [500000, -5000, 1000000],
        'score': np.random.normal(75, 10, 100).tolist() + [150, -20, 999],
        'category': np.random.choice(['A', 'B', 'C'], 103)
    })
    
    print("Dataset shape:", df.shape)
    print("\nDetecting outliers...\n")
    
    # Step 1: Detect outliers
    detector = OutlierDetector(df, output_file='potential_mistakes.csv')
    detector.detect_outliers(
        methods=['zscore', 'iqr', 'isolation'],
        z_threshold=3,
        iqr_multiplier=1.5
    )
    
    # Step 2: Review the CSV file 'potential_mistakes.csv'
    # Examine the flagged rows and decide which ones are actual mistakes
    
    # Step 3: Remove confirmed mistakes
    # After reviewing, specify which original_index values to remove
    indices_to_remove = [100, 101, 102]  # Example: these were confirmed as mistakes
    
    df_cleaned = detector.remove_outliers(indices_to_remove, df)
    
    # Save cleaned data
    # df_cleaned.to_csv('cleaned_data.csv', index=False)
    
    print("\nCleaning complete!")


# Example usage:
if __name__ == "__main__":
    ds, x, y = extract(label_column="GENHLTH",dataset_path="2021.csv")

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
        )
    


