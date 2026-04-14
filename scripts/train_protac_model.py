"""
PROTAC Activity Prediction Model Training
Using molecular descriptors and machine learning
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Descriptors as Desc
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import pickle
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_descriptors(smiles):
    """Calculate molecular descriptors from SMILES"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        descriptors = {
            'MW': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'HBA': Descriptors.NumHAcceptors(mol),
            'HBD': Descriptors.NumHDonors(mol),
            'RotBonds': Descriptors.NumRotatableBonds(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumRings': Descriptors.RingCount(mol),
            'NumAromaticRings': Descriptors.NumAromaticRings(mol),
            'NumHeavyAtoms': Descriptors.HeavyAtomCount(mol),
            'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
            'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
            'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
        }
        
        return descriptors
    except Exception as e:
        logger.debug(f"Error calculating descriptors: {e}")
        return None


def prepare_features(smiles_list):
    """Prepare features from SMILES list"""
    logger.info(f"Calculating descriptors for {len(smiles_list)} molecules...")
    
    descriptors_list = []
    valid_indices = []
    failed_count = 0
    
    for idx, smiles in enumerate(smiles_list):
        if idx % 1000 == 0:
            logger.info(f"Processing molecule {idx}/{len(smiles_list)}")
        
        try:
            descriptors = calculate_descriptors(smiles)
            if descriptors is not None:
                descriptors_list.append(descriptors)
                valid_indices.append(idx)
            else:
                failed_count += 1
        except Exception as e:
            failed_count += 1
    
    logger.info(f"Failed to process {failed_count} molecules")
    
    if not descriptors_list:
        raise ValueError("No valid molecules found")
    
    df_descriptors = pd.DataFrame(descriptors_list)
    logger.info(f"Successfully calculated descriptors for {len(descriptors_list)} molecules")
    
    return df_descriptors.values, np.array(valid_indices)


def train_model(train_csv, val_csv, test_csv, output_dir='models'):
    """Train and evaluate PROTAC activity prediction model"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    
    # Prepare features
    logger.info("\nPreparing training features...")
    X_train, train_valid_idx = prepare_features(train_df['PROTAC_SMILES'].values)
    y_train = train_df['Activity'].values[train_valid_idx]
    
    logger.info("\nPreparing validation features...")
    X_val, val_valid_idx = prepare_features(val_df['PROTAC_SMILES'].values)
    y_val = val_df['Activity'].values[val_valid_idx]
    
    logger.info("\nPreparing test features...")
    X_test, test_valid_idx = prepare_features(test_df['PROTAC_SMILES'].values)
    y_test = test_df['Activity'].values[test_valid_idx]
    
    # Scale features
    logger.info("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    results = {}
    
    for model_name, model_class in [('RandomForest', RandomForestClassifier), 
                                     ('GradientBoosting', GradientBoostingClassifier)]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_name} model...")
        logger.info(f"{'='*60}")
        
        if model_name == 'RandomForest':
            model = model_class(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        else:
            model = model_class(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                subsample=0.8
            )
        
        # Train
        logger.info("Training...")
        model.fit(X_train_scaled, y_train)
        
        # Evaluate on training set
        y_train_pred = model.predict(X_train_scaled)
        y_train_pred_proba = model.predict_proba(X_train_scaled)[:, 1]
        train_auc = roc_auc_score(y_train, y_train_pred_proba)
        train_acc = accuracy_score(y_train, y_train_pred)
        
        # Evaluate on validation set
        y_val_pred = model.predict(X_val_scaled)
        y_val_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_pred_proba)
        val_acc = accuracy_score(y_val, y_val_pred)
        
        # Evaluate on test set
        y_test_pred = model.predict(X_test_scaled)
        y_test_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        test_auc = roc_auc_score(y_test, y_test_pred_proba)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, zero_division=0)
        
        logger.info(f"\nTrain - AUC: {train_auc:.4f}, Accuracy: {train_acc:.4f}")
        logger.info(f"Val   - AUC: {val_auc:.4f}, Accuracy: {val_acc:.4f}")
        logger.info(f"Test  - AUC: {test_auc:.4f}, Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")
        
        # Save model
        model_path = output_path / f'{model_name.lower()}_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump({'model': model, 'scaler': scaler}, f)
        logger.info(f"Model saved to {model_path}")
        
        results[model_name] = {
            'train_auc': train_auc,
            'train_acc': train_acc,
            'val_auc': val_auc,
            'val_acc': val_acc,
            'test_auc': test_auc,
            'test_acc': test_acc,
            'test_f1': test_f1,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'predictions': y_test_pred_proba
        }
    
    # Save results summary
    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Train_AUC': [results[m]['train_auc'] for m in results.keys()],
        'Val_AUC': [results[m]['val_auc'] for m in results.keys()],
        'Test_AUC': [results[m]['test_auc'] for m in results.keys()],
        'Test_Accuracy': [results[m]['test_acc'] for m in results.keys()],
        'Test_F1': [results[m]['test_f1'] for m in results.keys()],
        'Test_Precision': [results[m]['test_precision'] for m in results.keys()],
        'Test_Recall': [results[m]['test_recall'] for m in results.keys()],
    })
    
    results_df.to_csv(output_path / 'model_results.csv', index=False)
    logger.info(f"\nModel Results:\n{results_df}")
    
    # Save predictions
    for model_name in results.keys():
        pred_df = pd.DataFrame({
            'True_Label': y_test,
            'Predicted_Probability': results[model_name]['predictions'],
            'Predicted_Label': (results[model_name]['predictions'] > 0.5).astype(int)
        })
        pred_df.to_csv(output_path / f'{model_name.lower()}_predictions.csv', index=False)
    
    return results


if __name__ == '__main__':
    train_csv = 'data/processed/train_data.csv'
    val_csv = 'data/processed/val_data.csv'
    test_csv = 'data/processed/test_data.csv'
    
    if Path(train_csv).exists() and Path(val_csv).exists() and Path(test_csv).exists():
        results = train_model(train_csv, val_csv, test_csv)
        logger.info("\nModel training completed successfully!")
    else:
        logger.error("Data files not found. Please run data preprocessing first.")
