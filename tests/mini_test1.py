import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Step 1: Identify categorical and numerical columns
categorical_cols = x_train.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = x_train.select_dtypes(include=[np.number]).columns.tolist()

print(f"Categorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

# Step 2: Handle categorical features with custom encoding that handles unseen categories
class SafeOneHotEncoder:
    """OneHotEncoder that creates a separate category for unseen values"""
    
    def __init__(self):
        self.encoders = {}
        self.feature_names = []
        
    def fit(self, X):
        """Fit on training data"""
        self.feature_names = []
        for col in X.columns:
            # Store unique values from training set
            unique_vals = X[col].fillna('__MISSING__').unique()
            self.encoders[col] = {
                'categories': set(unique_vals),
                'mapping': {val: idx for idx, val in enumerate(sorted(unique_vals))}
            }
            # Add feature names
            for val in sorted(unique_vals):
                self.feature_names.append(f"{col}_{val}")
            # Add unseen category feature name
            self.feature_names.append(f"{col}___UNSEEN__")
        return self
    
    def transform(self, X):
        """Transform data, handling unseen categories"""
        encoded_data = []
        
        for col in X.columns:
            col_data = X[col].fillna('__MISSING__')
            categories = self.encoders[col]['categories']
            
            # Create one-hot encoded matrix
            n_known = len(categories)
            encoded_col = np.zeros((len(X), n_known + 1))  # +1 for unseen
            
            for idx, val in enumerate(col_data):
                if val in categories:
                    # Known category
                    cat_idx = sorted(list(categories)).index(val)
                    encoded_col[idx, cat_idx] = 1
                else:
                    # Unseen category - use the last column
                    encoded_col[idx, -1] = 1
            
            encoded_data.append(encoded_col)
        
        # Concatenate all encoded columns
        result = np.hstack(encoded_data)
        return pd.DataFrame(result, columns=self.feature_names, index=X.index)
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        self.fit(X)
        return self.transform(X)

# Step 3: Encode categorical features
if categorical_cols:
    print("\n--- Encoding Categorical Features ---")
    cat_encoder = SafeOneHotEncoder()
    
    # Fit on training set and transform both train and test
    x_train_cat_encoded = cat_encoder.fit_transform(x_train[categorical_cols])
    x_test_cat_encoded = cat_encoder.transform(x_test[categorical_cols])
    
    print(f"Training set shape after encoding: {x_train_cat_encoded.shape}")
    print(f"Test set shape after encoding: {x_test_cat_encoded.shape}")
else:
    x_train_cat_encoded = pd.DataFrame(index=x_train.index)
    x_test_cat_encoded = pd.DataFrame(index=x_test.index)

# Step 4: Keep numerical features as-is (will scale later)
if numerical_cols:
    x_train_num = x_train[numerical_cols].copy()
    x_test_num = x_test[numerical_cols].copy()
else:
    x_train_num = pd.DataFrame(index=x_train.index)
    x_test_num = pd.DataFrame(index=x_test.index)

# Step 5: Combine encoded categorical and numerical features
x_train_combined = pd.concat([x_train_num, x_train_cat_encoded], axis=1)
x_test_combined = pd.concat([x_test_num, x_test_cat_encoded], axis=1)

print(f"\nCombined training set shape: {x_train_combined.shape}")
print(f"Combined test set shape: {x_test_combined.shape}")

# Step 6: Encode label column (y)
print("\n--- Encoding Labels ---")
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

print(f"Label classes: {label_encoder.classes_}")
print(f"Encoded training labels shape: {y_train_encoded.shape}")
print(f"Encoded test labels shape: {y_test_encoded.shape}")

# Step 7: Scale the features (fit on train, apply to test)
print("\n--- Scaling Features ---")
scaler = StandardScaler()

# Fit scaler on training data
x_train_scaled = scaler.fit_transform(x_train_combined)
x_test_scaled = scaler.transform(x_test_combined)

# Convert back to DataFrames
x_train_final = pd.DataFrame(
    x_train_scaled, 
    columns=x_train_combined.columns, 
    index=x_train.index
)
x_test_final = pd.DataFrame(
    x_test_scaled, 
    columns=x_test_combined.columns, 
    index=x_test.index
)

print(f"Final training set shape: {x_train_final.shape}")
print(f"Final test set shape: {x_test_final.shape}")

# Step 8: Summary
print("\n" + "="*50)
print("PREPROCESSING COMPLETE")
print("="*50)
print("\nFinal Variables:")
print(f"- x_train_final: {x_train_final.shape}")
print(f"- x_test_final: {x_test_final.shape}")
print(f"- y_train_encoded: {y_train_encoded.shape}")
print(f"- y_test_encoded: {y_test_encoded.shape}")
print("\nEncoders stored:")
print("- cat_encoder: For categorical features")
print("- label_encoder: For target labels")
print("- scaler: For feature scaling")