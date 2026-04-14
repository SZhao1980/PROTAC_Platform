"""
PROTAC Activity Prediction - Analysis and Visualization
Generates publication-quality figures and statistical analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score
from rdkit import Chem
from rdkit.Chem import Descriptors
import pickle
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ProtacAnalyzer:
    """Analyze and visualize PROTAC prediction results"""
    
    def __init__(self, output_dir='results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_roc_curves(self, predictions_dict, test_labels, output_file='roc_curves.png'):
        """Plot ROC curves for different models"""
        logger.info("Plotting ROC curves...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for idx, (model_name, predictions) in enumerate(predictions_dict.items()):
            fpr, tpr, _ = roc_curve(test_labels, predictions)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color=colors[idx], lw=2.5, 
                   label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        # Diagonal line
        ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
        
        ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
        ax.set_title('ROC Curves for PROTAC Activity Prediction', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved ROC curves to {self.output_dir / output_file}")
        plt.close()
    
    def plot_confusion_matrices(self, predictions_dict, test_labels, output_file='confusion_matrices.png'):
        """Plot confusion matrices for different models"""
        logger.info("Plotting confusion matrices...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        model_names = list(predictions_dict.keys())
        
        for idx, (model_name, predictions) in enumerate(predictions_dict.items()):
            pred_binary = (predictions > 0.5).astype(int)
            cm = confusion_matrix(test_labels, pred_binary)
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues', 
                       ax=axes[idx], cbar=False, square=True,
                       xticklabels=['Inactive', 'Active'],
                       yticklabels=['Inactive', 'Active'])
            
            axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label', fontsize=11, fontweight='bold')
            axes[idx].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrices to {self.output_dir / output_file}")
        plt.close()
    
    def plot_model_comparison(self, results_df, output_file='model_comparison.png'):
        """Plot model performance comparison"""
        logger.info("Plotting model comparison...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics = ['Train_AUC', 'Val_AUC', 'Test_AUC', 'Test_Accuracy']
        titles = ['Training AUC', 'Validation AUC', 'Test AUC', 'Test Accuracy']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]
            
            x_pos = np.arange(len(results_df))
            bars = ax.bar(x_pos, results_df[metric], color=['#1f77b4', '#ff7f0e'], 
                         alpha=0.8, edgecolor='black', linewidth=1.5)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.set_ylabel(title, fontsize=11, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(results_df['Model'], fontsize=10)
            ax.set_ylim([0, 1.1])
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved model comparison to {self.output_dir / output_file}")
        plt.close()
    
    def plot_descriptor_distribution(self, data_df, output_file='descriptor_distribution.png'):
        """Plot distribution of molecular descriptors"""
        logger.info("Plotting descriptor distributions...")
        
        descriptors = ['MW', 'LogP', 'TPSA', 'HBA', 'HBD', 'RotBonds']
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        for idx, descriptor in enumerate(descriptors):
            ax = axes[idx // 3, idx % 3]
            
            # Separate by activity
            active = data_df[data_df['Activity'] == 1][descriptor]
            inactive = data_df[data_df['Activity'] == 0][descriptor]
            
            ax.hist(active, bins=30, alpha=0.6, label='Active', color='#2ca02c', edgecolor='black')
            ax.hist(inactive, bins=30, alpha=0.6, label='Inactive', color='#d62728', edgecolor='black')
            
            ax.set_xlabel(descriptor, fontsize=11, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax.set_title(f'{descriptor} Distribution', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved descriptor distributions to {self.output_dir / output_file}")
        plt.close()
    
    def plot_prediction_distribution(self, predictions, test_labels, output_file='prediction_distribution.png'):
        """Plot distribution of predictions"""
        logger.info("Plotting prediction distributions...")
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        active_pred = predictions[test_labels == 1]
        inactive_pred = predictions[test_labels == 0]
        
        ax.hist(active_pred, bins=40, alpha=0.6, label='Active Compounds', 
               color='#2ca02c', edgecolor='black', linewidth=1.2)
        ax.hist(inactive_pred, bins=40, alpha=0.6, label='Inactive Compounds', 
               color='#d62728', edgecolor='black', linewidth=1.2)
        
        ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold')
        
        ax.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Predicted Probabilities', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved prediction distribution to {self.output_dir / output_file}")
        plt.close()
    
    def plot_target_distribution(self, data_df, output_file='target_distribution.png'):
        """Plot distribution of targets"""
        logger.info("Plotting target distribution...")
        
        target_counts = data_df['Target'].value_counts().head(15)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.barh(range(len(target_counts)), target_counts.values, 
                      color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=1.2)
        
        ax.set_yticks(range(len(target_counts)))
        ax.set_yticklabels(target_counts.index, fontsize=10)
        ax.set_xlabel('Number of PROTACs', fontsize=12, fontweight='bold')
        ax.set_title('Top 15 Target Proteins in Dataset', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, target_counts.values)):
            ax.text(value, i, f' {int(value)}', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved target distribution to {self.output_dir / output_file}")
        plt.close()
    
    def plot_e3_ligase_distribution(self, data_df, output_file='e3_ligase_distribution.png'):
        """Plot distribution of E3 ligases"""
        logger.info("Plotting E3 ligase distribution...")
        
        e3_counts = data_df['E3_Ligase'].value_counts().head(10)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        bars = ax.bar(range(len(e3_counts)), e3_counts.values, 
                     color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=1.2)
        
        ax.set_xticks(range(len(e3_counts)))
        ax.set_xticklabels(e3_counts.index, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Number of PROTACs', fontsize=12, fontweight='bold')
        ax.set_title('Top 10 E3 Ligases in Dataset', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars, e3_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2., value,
                   f'{int(value)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved E3 ligase distribution to {self.output_dir / output_file}")
        plt.close()
    
    def generate_summary_report(self, data_df, results_df, test_predictions, test_labels):
        """Generate comprehensive summary report"""
        logger.info("Generating summary report...")
        
        report = []
        report.append("="*80)
        report.append("PROTAC ACTIVITY PREDICTION - COMPREHENSIVE ANALYSIS REPORT")
        report.append("="*80)
        report.append("")
        
        # Dataset Statistics
        report.append("DATASET STATISTICS")
        report.append("-" * 80)
        report.append(f"Total Compounds: {len(data_df)}")
        report.append(f"Active Compounds: {(data_df['Activity'] == 1).sum()} ({(data_df['Activity'] == 1).sum()/len(data_df)*100:.1f}%)")
        report.append(f"Inactive Compounds: {(data_df['Activity'] == 0).sum()} ({(data_df['Activity'] == 0).sum()/len(data_df)*100:.1f}%)")
        report.append(f"Unique Targets: {data_df['Target'].nunique()}")
        report.append(f"Unique E3 Ligases: {data_df['E3_Ligase'].nunique()}")
        report.append("")
        
        # Molecular Properties
        report.append("MOLECULAR PROPERTIES (Mean ± Std)")
        report.append("-" * 80)
        for col in ['MW', 'LogP', 'TPSA', 'HBA', 'HBD', 'RotBonds']:
            if col in data_df.columns:
                mean = data_df[col].mean()
                std = data_df[col].std()
                report.append(f"{col:15s}: {mean:8.2f} ± {std:6.2f}")
        report.append("")
        
        # Model Performance
        report.append("MODEL PERFORMANCE")
        report.append("-" * 80)
        report.append(results_df.to_string(index=False))
        report.append("")
        
        # Best Model Analysis
        best_model_idx = results_df['Test_AUC'].idxmax()
        best_model = results_df.iloc[best_model_idx]
        report.append(f"Best Model: {best_model['Model']}")
        report.append(f"Test AUC: {best_model['Test_AUC']:.4f}")
        report.append(f"Test Accuracy: {best_model['Test_Accuracy']:.4f}")
        report.append(f"Test F1 Score: {best_model['Test_F1']:.4f}")
        report.append("")
        
        # Write report
        report_file = self.output_dir / 'analysis_report.txt'
        with open(report_file, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Saved report to {report_file}")
        
        return '\n'.join(report)


def main():
    """Main analysis and visualization pipeline"""
    
    analyzer = ProtacAnalyzer(output_dir='results')
    
    # Load data
    logger.info("Loading data...")
    merged_df = pd.read_csv('data/processed/merged_data.csv')
    test_df = pd.read_csv('data/processed/test_data.csv')
    results_df = pd.read_csv('models/model_results.csv')
    
    # Load predictions
    rf_pred = pd.read_csv('models/randomforest_predictions.csv')
    gb_pred = pd.read_csv('models/gradientboosting_predictions.csv')
    
    test_labels = rf_pred['True_Label'].values
    predictions_dict = {
        'Random Forest': rf_pred['Predicted_Probability'].values,
        'Gradient Boosting': gb_pred['Predicted_Probability'].values
    }
    
    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    
    # ROC curves
    analyzer.plot_roc_curves(predictions_dict, test_labels)
    
    # Confusion matrices
    analyzer.plot_confusion_matrices(predictions_dict, test_labels)
    
    # Model comparison
    analyzer.plot_model_comparison(results_df)
    
    # Descriptor distributions
    analyzer.plot_descriptor_distribution(merged_df)
    
    # Prediction distributions
    analyzer.plot_prediction_distribution(predictions_dict['Random Forest'], test_labels)
    
    # Target distribution
    analyzer.plot_target_distribution(merged_df)
    
    # E3 ligase distribution
    analyzer.plot_e3_ligase_distribution(merged_df)
    
    # Summary report
    report = analyzer.generate_summary_report(merged_df, results_df, 
                                             predictions_dict['Random Forest'], test_labels)
    print("\n" + report)
    
    logger.info("\nAnalysis and visualization completed!")


if __name__ == '__main__':
    main()
