from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold, RandomizedSearchCV
import numpy as np
from imblearn.over_sampling import SMOTE
import joblib
import os
import json
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# Constants
TARGET_MAP = {0: "zika", 1: "dengue", 2: "chikungunya"}  # Using lowercase to match database values

# Basic symptoms used for feature engineering
SYMPTOM_FEATURES = [
    "febre", "mialgia", "cefaleia", "exantema", "vomito",
    "nausea", "artralgia", "dor_costas", "conjuntvit", 
    "artrite", "leucopenia", "dor_retro", "petequia_n"
]

# All features for training (including engineered features that will be created)
BASE_FEATURES = SYMPTOM_FEATURES + [
    "articular_severity", "hemorrhagic_severity", "constitutional_severity",
    "zika_pattern", "dengue_pattern", "chik_pattern",
    "articular_constitutional_ratio", "hemorrhagic_articular_ratio",
    "exantema_hemorrhagic_ratio",
    "zika_specific_pattern", "dengue_specific_pattern", "chik_specific_pattern"
]

# Setup
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "model_artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def find_optimal_thresholds(y_true, y_pred_proba):
    """
    Find optimal prediction thresholds for each class using ROC curves
    """
    n_classes = y_pred_proba.shape[1]
    optimal_thresholds = []
    
    for i in range(n_classes):
        # Convert to binary classification problem
        y_true_binary = (y_true == i).astype(int)
        y_pred_proba_class = y_pred_proba[:, i]
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true_binary, y_pred_proba_class)
        
        # Find threshold that maximizes tpr - fpr (Youden's J statistic)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        optimal_thresholds.append(optimal_threshold)
    
    return np.array(optimal_thresholds)

def predict_with_thresholds(probas, thresholds):
    """
    Make predictions using class-specific thresholds
    """
    # Apply thresholds
    pred_with_threshold = (probas >= thresholds).astype(int)
    
    # Handle cases where no class or multiple classes are predicted
    row_sums = pred_with_threshold.sum(axis=1)
    
    # For rows with no prediction, take the highest probability
    no_pred = row_sums == 0
    if np.any(no_pred):
        pred_with_threshold[no_pred] = 0
        pred_with_threshold[no_pred, probas[no_pred].argmax(axis=1)] = 1
    
    # For rows with multiple predictions, take the highest probability
    multiple_pred = row_sums > 1
    if np.any(multiple_pred):
        pred_with_threshold[multiple_pred] = 0
        pred_with_threshold[multiple_pred, probas[multiple_pred].argmax(axis=1)] = 1
    
    return pred_with_threshold.argmax(axis=1)

def create_interaction_features(df):
    """Create interaction features for better disease discrimination."""
    print("\nCreating interaction features...")
    
    # 1. Base Severity Indices
    # Articular/Muscular Pattern (important for Chikungunya)
    df["articular_severity"] = (
        df["artralgia"] * 2.5 +  # Increased weight for joint pain
        df["artrite"] * 2.0 +    # Added arthritis
        df["mialgia"] * 1.5 +    # Muscle pain
        df["dor_costas"] * 1.0   # Back pain
    ) / 7.0
    
    # Hemorrhagic Pattern (important for Dengue)
    df["hemorrhagic_severity"] = (
        df["petequia_n"] * 2.0 +
        df["leucopenia"] * 1.5
    ) / 3.5
    
    # Constitutional Pattern (common in all but with variations)
    df["constitutional_severity"] = (
        df["febre"] * 2.0 +
        df["cefaleia"] * 1.5 +
        df["nausea"] * 1.0 +
        df["vomito"] * 1.0
    ) / 5.5
    
    # 2. Disease-Specific Patterns
    # Zika Pattern (enhanced with more specific weights)
    df["zika_pattern"] = (
        df["exantema"] * 4.0 +         # Increased weight for rash (key Zika symptom)
        df["conjuntvit"] * 3.5 +       # Increased weight for conjunctivitis
        df["artralgia"] * 1.0 +        # Mild joint pain
        df["febre"] * 0.8 -            # Fever less prominent
        df["hemorrhagic_severity"] * 2.0 -  # Stronger negative correlation with hemorrhagic
        df["articular_severity"] * 1.5    # Negative correlation with severe joint pain
    ) / 12.8  # Adjusted denominator
    
    # Disease-specific interaction patterns
    # Zika-specific pattern (refined)
    df["zika_specific_pattern"] = (
        (df["exantema"] * df["conjuntvit"]) * 3.0 +     # Strong co-occurrence of key Zika symptoms
        (1 - df["hemorrhagic_severity"]) * 2.0 +         # Absence of hemorrhagic symptoms
        (df["constitutional_severity"] < 0.5) * 1.5 +    # Mild constitutional symptoms
        (1 - df["articular_severity"]) * 1.5             # Mild joint symptoms
    ) / 8.0

    # Dengue-specific pattern
    df["dengue_specific_pattern"] = (
        (df["hemorrhagic_severity"] * df["constitutional_severity"]) * 3.0 +  # Strong hemorrhagic and constitutional
        (df["febre"] * df["dor_retro"]) * 2.0 +         # Fever with retroorbital pain
        (df["petequia_n"] * df["leucopenia"]) * 2.0 +   # Hemorrhagic manifestations
        (1 - df["articular_severity"]) * 1.0             # Less joint involvement
    ) / 8.0

    # Chikungunya-specific pattern
    df["chik_specific_pattern"] = (
        (df["articular_severity"] * df["artrite"]) * 3.0 +  # Strong joint involvement
        (df["febre"] * df["mialgia"]) * 2.0 +           # Fever with muscle pain
        (1 - df["hemorrhagic_severity"]) * 1.5 +        # Less hemorrhagic
        (df["constitutional_severity"] > 0.7) * 1.5      # Strong constitutional at onset
    ) / 8.0
    
    # Dengue Pattern (enhanced)
    df["dengue_pattern"] = (
        df["hemorrhagic_severity"] * 3.0 +     # Hemorrhagic signs very important
        df["constitutional_severity"] * 2.0 +   # Strong systemic symptoms
        df["dor_retro"] * 2.5 +                # Retroorbital pain highly specific
        df["febre"] * 1.5 +                    # High fever characteristic
        df["nausea"] * 1.2 +                   # GI symptoms
        df["vomito"] * 1.2 -                   # GI symptoms
        df["articular_severity"] * 1.0         # Less joint involvement
    ) / 12.4
    
    # Chikungunya Pattern (enhanced)
    df["chik_pattern"] = (
        df["articular_severity"] * 3.5 +        # Strong joint involvement is key
        df["artrite"] * 2.5 +                   # Arthritis very specific
        df["febre"] * 1.5 +                     # Sudden high fever
        df["mialgia"] * 1.5 +                   # Muscle pain
        df["constitutional_severity"] * 0.8 -    # Less systemic involvement
        df["hemorrhagic_severity"] * 1.2 -      # Negative correlation
        df["exantema"] * 0.8                    # Rash less common
    ) / 11.8
    
    # 3. Additional Interaction Terms
    # Ratios and interactions that help differentiation
    df["articular_constitutional_ratio"] = df["articular_severity"] / (df["constitutional_severity"] + 0.1)
    df["hemorrhagic_articular_ratio"] = df["hemorrhagic_severity"] / (df["articular_severity"] + 0.1)
    df["exantema_hemorrhagic_ratio"] = df["exantema"] / (df["hemorrhagic_severity"] + 0.1)
    
    # Normalize all engineered features to 0-1 range
    feature_cols = [col for col in df.columns if col.endswith('_severity') or 
                   col.endswith('_pattern') or col.endswith('_ratio')]
    
    scaler_minmax = MinMaxScaler()
    df[feature_cols] = scaler_minmax.fit_transform(df[feature_cols])
    
    return df

def main():
    """Main training pipeline with ensemble voting and threshold optimization."""
    try:
        # Connect to database
        engine = create_engine('postgresql://bisnet0:RG4J8^%*TWjA*977Y40T81B2@localhost:5432/episcope_db')
        print("Database connection initialized.")
        
        # Load data
        print("\nLoading data...")
        df = pd.read_sql("SELECT * FROM cleaned_arboviroses_cases", engine)
        print(f"Loaded {len(df)} records")
        
        # Check data
        if df.empty:
            raise ValueError("No data found in database")
        
        print("\nUnique values in 'doenca_alvo' and their counts:")
        print(df['doenca_alvo'].value_counts())
        
        # Map target variable (using lowercase values)
        df['target_encoded'] = df['doenca_alvo'].map({'zika': 0, 'dengue': 1, 'chikungunya': 2})
        
        print(f"\nNumber of non-null values in 'target_encoded' after mapping: {df['target_encoded'].notna().sum()}")
        print("\nMapping complete. Example of first few rows:")
        print(df[['doenca_alvo', 'target_encoded']].head())

        df.dropna(subset=['target_encoded'], inplace=True)
        df['target_encoded'] = df['target_encoded'].astype(int)

        # Create features
        print("\nPreparing features...")
        df = create_interaction_features(df)
        
        print("\nDataFrame columns after feature engineering:", df.columns.tolist())
        print("BASE_FEATURES before X assignment:", BASE_FEATURES)

        X = df[BASE_FEATURES]
        y = df['target_encoded']
        print(f"Feature matrix shape: {X.shape}")
        print("Features used for training (X.columns):", X.columns.tolist())
        
        # Split data
        print("\nSplitting data...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        print("Scaling features...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Balance data with careful SMOTE
        print("\nBalancing data with SMOTE...")
        # Use SMOTE with more conservative settings
        smote = SMOTE(
            random_state=42,
            k_neighbors=7,      # More neighbors for better synthetic samples
            sampling_strategy='auto'  # Automatically determine the sampling strategy
        )
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"Training set shape after SMOTE: {X_train.shape}")
        
        # Print class distribution after SMOTE
        unique, counts = np.unique(y_train, return_counts=True)
        print("\nClass distribution after SMOTE:")
        for label, count in zip(unique, counts):
            print(f"{TARGET_MAP[label]}: {count} samples")
        
        # Calculate class weights for better balance
        class_weights = {
            0: 1.5,  # Moderate boost for Zika (class 0)
            1: 1.2,  # Slight boost for Dengue (class 1)
            2: 1.0   # Base weight for Chikungunya (class 2)
        }

        # Train models with optimized parameters
        print("\nTraining models...")
        models = {
            "randomforest": RandomForestClassifier(
                n_estimators=1000,        # Increased number of trees
                max_depth=12,             # Moderate depth to prevent overfitting
                min_samples_split=10,     # More conservative splitting
                min_samples_leaf=4,       # More conservative leaf size
                max_features='sqrt',      # Standard for classification
                class_weight=class_weights,  # Custom class weights
                bootstrap=True,           # Enable bootstrapping
                max_samples=0.7,          # Sample size for each tree
                random_state=42,
                n_jobs=-1                 # Use all CPU cores
            ),
            "xgboost": XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                n_estimators=1200,         # More trees for better learning
                learning_rate=0.03,        # Lower learning rate
                max_depth=6,              # Reduced depth to prevent overfitting
                min_child_weight=5,       # More conservative splits
                gamma=0.3,                # Stronger regularization
                subsample=0.7,            # More aggressive subsampling
                colsample_bytree=0.7,     # More feature subsampling
                reg_alpha=0.2,            # Stronger L1 regularization
                reg_lambda=3.0,           # Stronger L2 regularization
                scale_pos_weight=1,
                random_state=42,
                use_label_encoder=False,
                eval_metric=['mlogloss', 'auc'],
                tree_method='hist',
                n_jobs=-1
            )
        }
        
        # Hyperparameter optimization settings
        param_distributions = {
            "randomforest": {
                "n_estimators": [800, 1000, 1200],
                "max_depth": [8, 10, 12, 15],
                "min_samples_split": [8, 10, 12],
                "min_samples_leaf": [3, 4, 5],
                "max_features": ["sqrt", "log2"],
            },
            "xgboost": {
                "n_estimators": [800, 1000, 1200],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.01, 0.1, 0.3],
                "subsample": [0.7, 0.8, 0.9],
                "colsample_bytree": [0.7, 0.8, 0.9],
                "min_child_weight": [3, 5, 7],
            }
        }

        # Train individual models with optimization
        optimized_models = {}
        for name, model in models.items():
            print(f"\nTraining {name.upper()}...")
            
            # Hyperparameter optimization
            print("\nPerforming hyperparameter optimization...")
            random_search = RandomizedSearchCV(
                model,
                param_distributions[name],
                n_iter=10,
                cv=5,
                scoring='balanced_accuracy',
                n_jobs=-1,
                random_state=42,
                verbose=1
            )
            random_search.fit(X_train, y_train)
            print(f"\nBest parameters found: {random_search.best_params_}")
            print(f"Best cross-validation score: {random_search.best_score_:.3f}")
            
            # Store optimized model
            optimized_models[name] = random_search.best_estimator_
        
        # Create and train voting classifier
        print("\nTraining Ensemble Voting Classifier...")
        voting_clf = VotingClassifier(
            estimators=[
                ('rf', optimized_models['randomforest']),
                ('xgb', optimized_models['xgboost'])
            ],
            voting='soft',  # Use probability estimates for voting
            weights=[1.2, 1.0]  # Give slightly more weight to RandomForest
        )
        voting_clf.fit(X_train, y_train)
        
        # Get probability predictions
        y_pred_proba = voting_clf.predict_proba(X_test)
        
        # Find optimal thresholds
        print("\nFinding optimal classification thresholds...")
        optimal_thresholds = find_optimal_thresholds(y_test, y_pred_proba)
        print("Optimal thresholds:", optimal_thresholds)
        
        # Make predictions using optimal thresholds
        y_pred = predict_with_thresholds(y_pred_proba, optimal_thresholds)
        
        print("\n=== ENSEMBLE VOTING Performance ===")
        print(classification_report(y_test, y_pred, target_names=list(TARGET_MAP.values())))
            
            # Perform stratified cross-validation for individual models
            print("Performing 5-fold stratified cross-validation...")
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(optimized_models[name], X_train, y_train, cv=skf, scoring='balanced_accuracy')
            print(f"Cross-validation scores: {cv_scores}")
            print(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            # Train final model
            print("\nTraining final model...")
            model.fit(X_train, y_train)
            
            # Feature Importance Analysis
            if name == "randomforest":
                importances = pd.DataFrame({
                    'feature': BASE_FEATURES,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                print("\nTop 10 Most Important Features:")
                print(importances.head(10))
            
            # Evaluate
            y_pred = model.predict(X_test)
            print(f"\n=== {name.upper()} Performance ===")
            print(classification_report(y_test, y_pred, target_names=list(TARGET_MAP.values())))
            
            # Plot confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            print("\nConfusion Matrix:")
            print(cm)
            print("\nRows: True labels")
            print("Columns: Predicted labels")
            print("Order: Zika, Dengue, Chikungunya")
            
            # Save individual models
            model_path = os.path.join(ARTIFACTS_DIR, f"{name}_model.joblib")
            joblib.dump(optimized_models[name], model_path)
            print(f"Saved {name} model to {model_path}")
        
        # Save ensemble model
        ensemble_path = os.path.join(ARTIFACTS_DIR, "ensemble_model.joblib")
        joblib.dump(voting_clf, ensemble_path)
        print(f"\nSaved ensemble model to {ensemble_path}")
        
        # Save thresholds
        thresholds_path = os.path.join(ARTIFACTS_DIR, "optimal_thresholds.joblib")
        joblib.dump(optimal_thresholds, thresholds_path)
        print(f"Saved optimal thresholds to {thresholds_path}")
        
        # Save artifacts
        print("\nSaving additional artifacts...")
        joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "scaler.joblib"))
        print(f"Saved scaler.")
        
        with open(os.path.join(ARTIFACTS_DIR, "model_columns.json"), "w") as f:
            json.dump(BASE_FEATURES, f)
        print("Saved model columns.")

        with open(os.path.join(ARTIFACTS_DIR, "target_map.json"), "w") as f:
            json.dump(TARGET_MAP, f)
        print("Saved target map.")
        
        print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        raise
    finally:
        if 'engine' in locals() and engine:
            engine.dispose()

if __name__ == "__main__":
    main()