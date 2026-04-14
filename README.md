# PROTAC_Platform
This project implements an AI-powered PROTAC (Proteolysis Targeting Chimeras) discovery platform using machine learning and graph neural networks for predicting PROTAC activity and designing novel degraders.

A machine learning-based framework for predicting PROTAC degradation activity using molecular descriptors and structural features.

This project integrates data preprocessing, statistical analysis, and predictive modeling to explore structure–activity relationships in PROTAC compounds.

## Project Structure

```
PROTAC_AI_Platform/
├── data/
│   ├── raw/                 # Original datasets
│   └── processed/           # Preprocessed data (train/val/test splits)
├── models/                  # Trained models and predictions
├── results/                 # Analysis results and visualizations
├── logs/                    # Training and processing logs
└── scripts/
     ├── data_preprocessing.py    # Data loading and preprocessing
     ├── train_protac_model.py    # Model training script
     ├── analysis_and_visualization.py  # Analysis and figure generation
     ├── gnn_model.py            # Graph neural network implementation
     ├──simple_model.py         # Simplified model implementations
     └── ....
```

## Usage Instructions

1. **Data Preprocessing**:
   ```bash
   python3 data_preprocessing.py
   ```

2. **Model Training**:
   ```bash
   python3 train_protac_model.py
   ```

3. **Analysis and Visualization**:
   ```bash
   python3 analysis_and_visualization.py
   ```
## Key Achievements

### 1. Data Processing
- **Total Compounds Processed**: 10,583
- **Active Compounds**: 10,192 (96.3%)
- **Inactive Compounds**: 391 (3.7%)
- **Unique Targets**: 521
- **E3 Ligase Types**: 25

### 2. Molecular Properties (Mean ± Std)
- Molecular Weight: 946.68 ± 171.98 Da
- LogP: 5.42 ± 2.37
- TPSA: 215.62 ± 44.60 Ų
- H-Bond Acceptors: 13.40 ± 3.01
- H-Bond Donors: 4.33 ± 1.67
- Rotatable Bonds: 18.72 ± 6.45

### 3. Model Performance

#### Random Forest Model (Best Performer)
- Training AUC: 0.9764
- Validation AUC: 0.7907
- Test AUC: 0.7892
- Test Accuracy: 95.15%
- Test F1 Score: 0.9752
- Test Precision: 0.9533
- Test Recall: 0.9980

#### Gradient Boosting Model
- Training AUC: 0.9711
- Validation AUC: 0.7679
- Test AUC: 0.7352
- Test Accuracy: 94.96%
- Test F1 Score: 0.9741

### 4. Generated Figures (Publication Quality)
1. ROC Curves - Model performance comparison
2. Model Comparison - Training, validation, and test metrics
3. Descriptor Distribution - Molecular property distributions
4. Prediction Distribution - Predicted probability distributions
5. Target Distribution - Top 15 target proteins
6. E3 Ligase Distribution - Top 10 E3 ligases
7. Confusion Matrices - Classification performance

## Reference

**Title:** _Machine Learning-Driven Prediction of PROTAC Degradation Activity: A Large-Scale Structural and Molecular Property Analysis_
**Author:** Sai Zhao、Yuxuan Wu、Yuanzhen Cai
**Email:** [z_s_1980@hotmail.com](mailto:z_s_1980@hotmail.com)
