from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
import joblib
import os
import json
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# Constants
TARGET_MAP = {0: "zika", 1: "dengue", 2: "chikungunya"}

# Basic symptoms used for feature engineering
SYMPTOM_FEATURES = [
    "febre", "mialgia", "cefaleia", "exantema", "vomito",
    "nausea", "artralgia", "dor_costas", "conjuntvit", 
    "artrite", "leucopenia", "dor_retro", "petequia_n"
]

# All features for training
BASE_FEATURES = SYMPTOM_FEATURES + [
    "articular_severity", "hemorrhagic_severity", "constitutional_severity",
    "zika_pattern", "dengue_pattern", "chik_pattern",
    "articular_constitutional_ratio", "hemorrhagic_articular_ratio",
    "exantema_hemorrhagic_ratio", 
    "zika_primary_pattern", "zika_secondary_pattern", "zika_specific_pattern",
    "dengue_specific_pattern", "chik_specific_pattern"
]

# Setup
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "model_artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def find_optimal_thresholds(y_true, y_pred_proba, class_weights=None):
    """Find optimal prediction thresholds for each class using ROC curves with class-specific weights"""
    n_classes = y_pred_proba.shape[1]
    optimal_thresholds = []
    
    if class_weights is None:
        class_weights = {0: 1.5, 1: 1.0, 2: 1.0}  # Default weights
    
    for i in range(n_classes):
        y_true_binary = (y_true == i).astype(int)
        y_pred_proba_class = y_pred_proba[:, i]
        fpr, tpr, thresholds = roc_curve(y_true_binary, y_pred_proba_class)
        
        # Adjust threshold selection based on class
        if i == 0:  # Zika
            # For Zika, prioritize recall by using a lower threshold
            # Use the threshold that gives at least 80% TPR
            target_tpr = 0.8
            valid_idx = np.where(tpr >= target_tpr)[0]
            if len(valid_idx) > 0:
                optimal_idx = valid_idx[np.argmax(tpr[valid_idx] - fpr[valid_idx])]
            else:
                optimal_idx = np.argmax(tpr - fpr)
        elif i == 2:  # Chikungunya
            # For Chikungunya, prioritize precision by using a higher threshold
            # Use the threshold that gives at most 20% FPR
            target_fpr = 0.2
            valid_idx = np.where(fpr <= target_fpr)[0]
            if len(valid_idx) > 0:
                optimal_idx = valid_idx[np.argmax(tpr[valid_idx] - fpr[valid_idx])]
            else:
                optimal_idx = np.argmax(tpr - fpr)
        else:  # Dengue
            # For Dengue, use standard threshold optimization
            optimal_idx = np.argmax((tpr - fpr) * class_weights[i])
        
        optimal_threshold = thresholds[optimal_idx]
        optimal_thresholds.append(optimal_threshold)
    
    return np.array(optimal_thresholds)

def predict_with_thresholds(probas, thresholds):
    """Make predictions using class-specific thresholds"""
    pred_with_threshold = (probas >= thresholds).astype(int)
    row_sums = pred_with_threshold.sum(axis=1)
    
    no_pred = row_sums == 0
    if np.any(no_pred):
        pred_with_threshold[no_pred] = 0
        pred_with_threshold[no_pred, probas[no_pred].argmax(axis=1)] = 1
    
    multiple_pred = row_sums > 1
    if np.any(multiple_pred):
        pred_with_threshold[multiple_pred] = 0
        pred_with_threshold[multiple_pred, probas[multiple_pred].argmax(axis=1)] = 1
    
    return pred_with_threshold.argmax(axis=1)

def create_interaction_features(df):
    """Create interaction features for better disease discrimination."""
    print("\nCreating interaction features...")
    
    # Base severity indices
    df["articular_severity"] = (
        df["artralgia"] * 2.5 +
        df["artrite"] * 2.0 +
        df["mialgia"] * 1.5 +
        df["dor_costas"] * 1.0
    ) / 7.0
    
    df["hemorrhagic_severity"] = (
        df["petequia_n"] * 2.0 +
        df["leucopenia"] * 1.5
    ) / 3.5
    
    df["constitutional_severity"] = (
        df["febre"] * 2.0 +
        df["cefaleia"] * 1.5 +
        df["nausea"] * 1.0 +
        df["vomito"] * 1.0
    ) / 5.5
    
    # Disease-specific patterns
    df["zika_pattern"] = (
        df["exantema"] * 4.0 +
        df["conjuntvit"] * 3.5 +
        df["artralgia"] * 1.0 +
        df["febre"] * 0.8 -
        df["hemorrhagic_severity"] * 2.0 -
        df["articular_severity"] * 1.5
    ) / 12.8
    
    # Disease-specific interaction patterns
    df["zika_specific_pattern"] = (
        (df["exantema"] * df["conjuntvit"]) * 3.0 +
        (1 - df["hemorrhagic_severity"]) * 2.0 +
        (df["constitutional_severity"] < 0.5) * 1.5 +
        (1 - df["articular_severity"]) * 1.5
    ) / 8.0

    df["dengue_specific_pattern"] = (
        (df["hemorrhagic_severity"] * df["constitutional_severity"]) * 3.0 +
        (df["febre"] * df["dor_retro"]) * 2.0 +
        (df["petequia_n"] * df["leucopenia"]) * 2.0 +
        (1 - df["articular_severity"]) * 1.0
    ) / 8.0

    df["chik_specific_pattern"] = (
        (df["articular_severity"] * df["artrite"]) * 3.0 +
        (df["febre"] * df["mialgia"]) * 2.0 +
        (1 - df["hemorrhagic_severity"]) * 1.5 +
        (df["constitutional_severity"] > 0.7) * 1.5
    ) / 8.0
    
    df["dengue_pattern"] = (
        df["hemorrhagic_severity"] * 3.0 +
        df["constitutional_severity"] * 2.0 +
        df["dor_retro"] * 2.5 +
        df["febre"] * 1.5 +
        df["nausea"] * 1.2 +
        df["vomito"] * 1.2 -
        df["articular_severity"] * 1.0
    ) / 12.4
    
    df["chik_pattern"] = (
        df["articular_severity"] * 3.5 +
        df["artrite"] * 2.5 +
        df["febre"] * 1.5 +
        df["mialgia"] * 1.5 +
        df["constitutional_severity"] * 0.8 -
        df["hemorrhagic_severity"] * 1.2 -
        df["exantema"] * 0.8
    ) / 11.8
    
    # Additional ratios
    df["articular_constitutional_ratio"] = df["articular_severity"] / (df["constitutional_severity"] + 0.1)
    df["hemorrhagic_articular_ratio"] = df["hemorrhagic_severity"] / (df["articular_severity"] + 0.1)
    df["exantema_hemorrhagic_ratio"] = df["exantema"] / (df["hemorrhagic_severity"] + 0.1)
    
    # Normalize engineered features
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
        
        # Map target variable
        df['target_encoded'] = df['doenca_alvo'].map({'zika': 0, 'dengue': 1, 'chikungunya': 2})
        
        print(f"\nNumber of non-null values in 'target_encoded': {df['target_encoded'].notna().sum()}")
        print("\nMapping complete. Example of first few rows:")
        print(df[['doenca_alvo', 'target_encoded']].head())

        df.dropna(subset=['target_encoded'], inplace=True)
        df['target_encoded'] = df['target_encoded'].astype(int)

        # Create features
        df = create_interaction_features(df)
        
        X = df[BASE_FEATURES]
        y = df['target_encoded']
        print(f"Feature matrix shape: {X.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Balance data with SMOTE
        print("\nBalancing data with SMOTE...")
        smote = SMOTE(random_state=42, k_neighbors=7)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        
        # Model configurations
        models = {
            "randomforest": RandomForestClassifier(
                n_estimators=1000,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=4,
                max_features='sqrt',
                class_weight='balanced_subsample',
                bootstrap=True,
                max_samples=0.7,
                random_state=42,
                n_jobs=-1
            ),
            "xgboost": XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                n_estimators=1200,
                learning_rate=0.03,
                max_depth=6,
                min_child_weight=5,
                gamma=0.3,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=0.2,
                reg_lambda=3.0,
                random_state=42,
                use_label_encoder=False,
                eval_metric=['mlogloss', 'auc'],
                tree_method='hist',
                n_jobs=-1
            )
        }

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
                "learning_rate": [0.01, 0.03, 0.1],
                "subsample": [0.7, 0.8, 0.9],
                "colsample_bytree": [0.7, 0.8, 0.9],
                "min_child_weight": [3, 5, 7],
            }
        }

        # Train individual models
        optimized_models = {}
        for name, model in models.items():
            print(f"\nOptimizing {name.upper()}...")
            
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
            print(f"Best parameters found: {random_search.best_params_}")
            print(f"Best cross-validation score: {random_search.best_score_:.3f}")
            
            optimized_models[name] = random_search.best_estimator_
            
            print(f"\nEvaluating {name.upper()}...")
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(optimized_models[name], X_train, y_train, cv=skf, scoring='balanced_accuracy')
            print(f"Cross-validation scores: {cv_scores}")
            print(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

        # Create Zika specialist model
        print("\nTraining Zika Specialist Model...")
        zika_specialist = RandomForestClassifier(
            n_estimators=1500,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=3,
            max_features='sqrt',
            class_weight={0: 2.0, 1: 0.5, 2: 0.5},  # Strong focus on Zika
            bootstrap=True,
            max_samples=0.8,
            random_state=42,
            n_jobs=-1
        )
        zika_specialist.fit(X_train, y_train)
        
        # Create and train voting classifier with Zika specialist
        print("\nTraining Enhanced Ensemble Voting Classifier...")
        voting_clf = VotingClassifier(
            estimators=[
                ('rf', optimized_models['randomforest']),
                ('xgb', optimized_models['xgboost']),
                ('zika_specialist', zika_specialist)
            ],
            voting='soft',
            weights=[1.0, 1.0, 1.5]  # Give more weight to Zika specialist
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
        
        # Evaluate ensemble
        print("\n=== ENSEMBLE VOTING Performance ===")
        print(classification_report(y_test, y_pred, target_names=list(TARGET_MAP.values())))
        
        # Show confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        print("\nRows: True labels")
        print("Columns: Predicted labels")
        print("Order: Zika, Dengue, Chikungunya")

        # Save models and artifacts
        print("\nSaving models and artifacts...")
        for name, model in optimized_models.items():
            model_path = os.path.join(ARTIFACTS_DIR, f"{name}_model.joblib")
            joblib.dump(model, model_path)
            print(f"Saved {name} model to {model_path}")
        
        ensemble_path = os.path.join(ARTIFACTS_DIR, "ensemble_model.joblib")
        joblib.dump(voting_clf, ensemble_path)
        print(f"Saved ensemble model to {ensemble_path}")
        
        thresholds_path = os.path.join(ARTIFACTS_DIR, "optimal_thresholds.joblib")
        joblib.dump(optimal_thresholds, thresholds_path)
        print(f"Saved optimal thresholds to {thresholds_path}")
        
        joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "scaler.joblib"))
        print("Saved scaler.")
        
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