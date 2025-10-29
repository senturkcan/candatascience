from keras.datasets import mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class MLOutlierDetector:
    """
    Detects potential mislabeled instances and extreme feature values
    for machine learning classification tasks.
    """
    
    def __init__(self, X, y, output_file='potential_mistakes.csv'):
        """
        Initialize the detector.
        
        Args:
            X: Features DataFrame
            y: Labels Series/DataFrame
            output_file: CSV filename to save potential mistakes
        """
        self.X = X.copy()
        self.y = y.copy() if isinstance(y, pd.Series) else y.copy().squeeze()
        self.X['original_index'] = X.index
        self.output_file = output_file
        
    def detect_mislabeled(self, method='knn', n_neighbors=5, contamination=0.1):
        """
        Detect potentially mislabeled instances - MAIN FUNCTIONALITY.
        
        Args:
            method: 'knn' (K-Nearest Neighbors) or 'isolation' (Isolation Forest per class)
            n_neighbors: Number of neighbors to check (for KNN method)
            contamination: Expected proportion of outliers (for isolation method)
        
        Returns:
            DataFrame with suspected mislabeled instances
        """
        print(f"\n{'='*60}")
        print("DETECTING POTENTIALLY MISLABELED INSTANCES")
        print(f"{'='*60}\n")
        
        results = []
        numeric_cols = self.X.select_dtypes(include=[np.number]).columns.tolist()
        if 'original_index' in numeric_cols:
            numeric_cols.remove('original_index')
        
        if method == 'knn':
            print(f"Using KNN method with {n_neighbors} neighbors...")
            from sklearn.neighbors import NearestNeighbors
            
            # Standardize features
            X_std = (self.X[numeric_cols] - self.X[numeric_cols].mean()) / self.X[numeric_cols].std()
            X_std = X_std.fillna(0)
            
            # Find nearest neighbors
            knn = NearestNeighbors(n_neighbors=n_neighbors + 1)
            knn.fit(X_std)
            distances, indices = knn.kneighbors(X_std)
            
            # Check label consistency with neighbors
            for i in range(len(self.X)):
                neighbor_indices = indices[i][1:]  # Exclude self
                neighbor_labels = self.y.iloc[neighbor_indices]
                current_label = self.y.iloc[i]
                
                # Count how many neighbors have different labels
                different_labels = (neighbor_labels != current_label).sum()
                agreement_ratio = 1 - (different_labels / n_neighbors)
                
                # Flag if minority among neighbors
                if different_labels >= n_neighbors * 0.6:  # 60% or more disagree
                    neighbor_label_counts = neighbor_labels.value_counts()
                    suggested_label = neighbor_label_counts.index[0]
                    
                    results.append({
                        'original_index': self.X.iloc[i]['original_index'],
                        'current_label': current_label,
                        'suggested_label': suggested_label,
                        'neighbor_agreement': f"{agreement_ratio:.1%}",
                        'neighbors_with_different_label': different_labels,
                        'avg_distance_to_neighbors': distances[i][1:].mean(),
                        'confidence': 'HIGH' if different_labels >= n_neighbors * 0.8 else 'MEDIUM',
                        'reason': f'{different_labels}/{n_neighbors} neighbors have different labels'
                    })
        
        elif method == 'isolation':
            print(f"Using Isolation Forest per class (contamination={contamination})...")
            from sklearn.ensemble import IsolationForest
            
            X_std = (self.X[numeric_cols] - self.X[numeric_cols].mean()) / self.X[numeric_cols].std()
            X_std = X_std.fillna(0)
            
            # Check each class separately
            for label in self.y.unique():
                class_mask = self.y == label
                class_indices = self.X[class_mask].index
                X_class = X_std[class_mask]
                
                if len(X_class) < 3:
                    continue
                
                iso = IsolationForest(contamination=min(contamination, 0.5), random_state=42)
                predictions = iso.fit_predict(X_class)
                
                outlier_mask = predictions == -1
                outlier_indices = X_class[outlier_mask].index
                
                for idx in outlier_indices:
                    loc = self.X.index.get_loc(idx)
                    results.append({
                        'original_index': self.X.iloc[loc]['original_index'],
                        'current_label': label,
                        'suggested_label': 'REVIEW',
                        'neighbor_agreement': 'N/A',
                        'neighbors_with_different_label': 'N/A',
                        'avg_distance_to_neighbors': 'N/A',
                        'confidence': 'MEDIUM',
                        'reason': f'Anomalous features for class {label}'
                    })
        
        if results:
            mislabeled_df = pd.DataFrame(results)
            
            # Add feature values for context
            for idx in mislabeled_df['original_index']:
                row_loc = self.X[self.X['original_index'] == idx].index[0]
                for col in numeric_cols[:5]:  # Add first 5 features
                    mislabeled_df.loc[mislabeled_df['original_index'] == idx, col] = self.X.loc[row_loc, col]
            
            mislabeled_df = mislabeled_df.sort_values('confidence', ascending=False)
            
            print(f"\n✓ Found {len(mislabeled_df)} potentially mislabeled instances")
            print(f"\nTop suspected mislabels:")
            print(mislabeled_df[['original_index', 'current_label', 'suggested_label', 'confidence', 'reason']].head(10).to_string(index=False))
            
            return mislabeled_df
        else:
            print("No mislabeled instances detected!")
            return pd.DataFrame()
    
    def detect_extreme_features(self, z_threshold=4, iqr_multiplier=3):
        """
        Detect extreme feature values - SECONDARY FUNCTIONALITY.
        Uses simple statistical methods.
        
        Args:
            z_threshold: Z-score threshold (higher = only very extreme values)
            iqr_multiplier: IQR multiplier (higher = only very extreme values)
        
        Returns:
            DataFrame with extreme feature values
        """
        print(f"\n{'='*60}")
        print("DETECTING EXTREME FEATURE VALUES")
        print(f"{'='*60}\n")
        
        results = []
        numeric_cols = self.X.select_dtypes(include=[np.number]).columns.tolist()
        if 'original_index' in numeric_cols:
            numeric_cols.remove('original_index')
        
        print(f"Using Z-Score (threshold={z_threshold}) and IQR (multiplier={iqr_multiplier})...\n")
        
        for col in numeric_cols:
            col_data = self.X[col].dropna()
            
            # Z-Score method
            z_scores = np.abs(stats.zscore(col_data))
            z_outliers = z_scores > z_threshold
            
            # IQR method
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - iqr_multiplier * IQR
            upper = Q3 + iqr_multiplier * IQR
            iqr_outliers = (col_data < lower) | (col_data > upper)
            
            # Combine both methods
            combined_outliers = z_outliers | iqr_outliers
            outlier_indices = col_data[combined_outliers].index
            
            for idx in outlier_indices:
                value = self.X.loc[idx, col]
                z_score = z_scores[col_data.index.get_loc(idx)]
                
                results.append({
                    'original_index': self.X.loc[idx, 'original_index'],
                    'feature': col,
                    'value': value,
                    'z_score': f"{z_score:.2f}",
                    'iqr_range': f"[{lower:.2f}, {upper:.2f}]",
                    'label': self.y.loc[idx],
                    'severity': 'EXTREME' if z_score > z_threshold * 1.5 else 'HIGH',
                    'reason': f'Z-score={z_score:.2f}, outside IQR range'
                })
        
        if results:
            extreme_df = pd.DataFrame(results)
            extreme_df = extreme_df.sort_values('z_score', ascending=False)
            
            print(f"✓ Found {len(extreme_df)} extreme feature values across {extreme_df['original_index'].nunique()} rows")
            print(f"\nTop extreme values:")
            print(extreme_df[['original_index', 'feature', 'value', 'z_score', 'severity']].head(10).to_string(index=False))
            
            return extreme_df
        else:
            print("No extreme feature values detected!")
            return pd.DataFrame()
    
    def detect_all(self, save_separate=True, **kwargs):
        """
        Run both detection methods and save results.
        
        Args:
            save_separate: If True, save mislabels and extremes in separate CSVs
            **kwargs: Arguments for detection methods
        """
        # Detect mislabeled (MAIN)
        mislabeled = self.detect_mislabeled(
            method=kwargs.get('mislabel_method', 'knn'),
            n_neighbors=kwargs.get('n_neighbors', 5),
            contamination=kwargs.get('contamination', 0.1)
        )
        
        # Detect extreme features (SECONDARY)
        extreme = self.detect_extreme_features(
            z_threshold=kwargs.get('z_threshold', 4),
            iqr_multiplier=kwargs.get('iqr_multiplier', 3)
        )
        
        # Save results
        if save_separate:
            if not mislabeled.empty:
                mislabeled.to_csv('mislabeled_instances.csv', index=False)
                print(f"\n✓ Saved to 'mislabeled_instances.csv'")
            
            if not extreme.empty:
                extreme.to_csv('extreme_features.csv', index=False)
                print(f"✓ Saved to 'extreme_features.csv'")
        else:
            # Combine both
            if not mislabeled.empty or not extreme.empty:
                mislabeled['type'] = 'MISLABEL'
                extreme['type'] = 'EXTREME_FEATURE'
                combined = pd.concat([mislabeled, extreme], ignore_index=True)
                combined.to_csv(self.output_file, index=False)
                print(f"\n✓ Saved to '{self.output_file}'")
        
        return mislabeled, extreme
    
    @staticmethod
    def remove_rows(X, y, indices_to_remove):
        """
        Remove confirmed mistakes after manual review.
        
        Args:
            X: Features DataFrame
            y: Labels Series/DataFrame
            indices_to_remove: List of row indices to remove
            
        Returns:
            Cleaned X and y
        """
        X_clean = X.drop(indices_to_remove, errors='ignore')
        y_clean = y.drop(indices_to_remove, errors='ignore')
        
        removed = len(X) - len(X_clean)
        print(f"\n✓ Removed {removed} rows")
        print(f"  X: {len(X)} → {len(X_clean)}")
        print(f"  y: {len(y)} → {len(y_clean)}")
        
        return X_clean, y_clean


# ============= USAGE EXAMPLE =============

# STEP 1: DETECT MISLABELED INSTANCES (MAIN FUNCTIONALITY)
detector = MLOutlierDetector(x, y)

# Option A: Only detect mislabeled instances
mislabeled = detector.detect_mislabeled(method='knn', n_neighbors=5)
mislabeled.to_csv('mislabeled_instances.csv', index=False)

# Option B: Detect both mislabeled + extreme features
mislabeled, extreme = detector.detect_all(
    save_separate=True,          # Save in separate CSV files
    mislabel_method='knn',       # 'knn' or 'isolation'
    n_neighbors=5,               # For KNN
    z_threshold=4,               # For extreme features (higher = stricter)
    iqr_multiplier=3             # For extreme features (higher = stricter)
)


# STEP 2: MANUAL REVIEW
# Review 'mislabeled_instances.csv' and 'extreme_features.csv'
# Identify actual mistakes


# STEP 3: REMOVE CONFIRMED MISTAKES
# After reviewing, use row indices to remove them
x_clean, y_clean = MLOutlierDetector.remove_rows(x, y, [45, 67, 89, 103])

# Save cleaned data
# x_clean.to_csv('x_cleaned.csv', index=False)
# y_clean.to_csv('y_cleaned.csv', index=False)