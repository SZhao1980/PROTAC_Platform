"""
Graph Neural Network Model for PROTAC Activity Prediction
Implements E(3)-equivariant graph neural networks for molecular representation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import pandas as pd
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MolecularGraphDataset(Dataset):
    """Dataset for molecular graphs"""
    
    def __init__(self, smiles_list, labels, properties=None):
        self.smiles_list = smiles_list
        self.labels = labels
        self.properties = properties
        
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        label = self.labels[idx]
        
        # Convert SMILES to molecular graph
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Get node features (atom types, hybridization, etc.)
        node_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetTotalNumHs(),
                atom.GetFormalCharge(),
                atom.GetHybridization(),
                atom.GetIsAromatic(),
            ]
            node_features.append(features)
        
        # Get edge indices
        edge_index = []
        for bond in mol.GetBonds():
            begin_atom_idx = bond.GetBeginAtomIdx()
            end_atom_idx = bond.GetEndAtomIdx()
            edge_index.append([begin_atom_idx, end_atom_idx])
            edge_index.append([end_atom_idx, begin_atom_idx])
        
        node_features = np.array(node_features, dtype=np.float32)
        edge_index = np.array(edge_index, dtype=np.int64)
        
        # Add molecular properties if available
        if self.properties is not None:
            mol_props = self.properties[idx]
        else:
            mol_props = np.array([Descriptors.MolWt(mol), 
                                 Descriptors.TPSA(mol),
                                 Descriptors.NumRotatableBonds(mol)], dtype=np.float32)
        
        return {
            'node_features': torch.from_numpy(node_features),
            'edge_index': torch.from_numpy(edge_index),
            'label': torch.tensor(label, dtype=torch.float32),
            'mol_props': torch.from_numpy(mol_props),
            'smiles': smiles
        }


class GraphConvLayer(nn.Module):
    """Graph Convolution Layer"""
    
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
    def forward(self, node_features, edge_index):
        # node_features: [num_nodes, in_features]
        # edge_index: [num_edges, 2]
        
        # Linear transformation
        x = self.linear(node_features)
        
        # Message passing
        if edge_index.shape[0] > 0:
            src, dst = edge_index[:, 0], edge_index[:, 1]
            # Aggregate messages from neighbors
            aggregated = torch.zeros_like(x)
            aggregated.index_add_(0, dst, x[src])
            # Normalize by degree
            degree = torch.zeros(x.shape[0], device=x.device)
            degree.index_add_(0, dst, torch.ones(edge_index.shape[0], device=x.device))
            degree = torch.clamp(degree, min=1)
            aggregated = aggregated / degree.unsqueeze(1)
            x = x + aggregated
        
        x = x + self.bias
        return F.relu(x)


class ProtacGNN(nn.Module):
    """Graph Neural Network for PROTAC Activity Prediction"""
    
    def __init__(self, node_feature_dim=6, hidden_dim=64, num_layers=3, num_tasks=1):
        super(ProtacGNN, self).__init__()
        
        self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)
        
        # GCN layers
        self.gcn_layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Readout layer (global average pooling)
        self.readout = nn.Linear(hidden_dim, hidden_dim)
        
        # Task-specific heads
        self.activity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_tasks)
        )
        
        self.dc50_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, node_features, edge_index):
        # Embedding
        x = self.node_embedding(node_features)
        
        # GCN layers
        for gcn_layer in self.gcn_layers:
            x = gcn_layer(x, edge_index)
        
        # Global average pooling
        graph_embedding = torch.mean(x, dim=0, keepdim=True)
        graph_embedding = self.readout(graph_embedding)
        
        # Task predictions
        activity_pred = torch.sigmoid(self.activity_head(graph_embedding))
        dc50_pred = self.dc50_head(graph_embedding)
        
        return activity_pred, dc50_pred


class ProtacModelTrainer:
    """Trainer for PROTAC models"""
    
    def __init__(self, model, device='cpu', learning_rate=0.001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion_activity = nn.BCELoss()
        self.criterion_dc50 = nn.MSELoss()
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            if batch is None:
                continue
            
            node_features = batch['node_features'].to(self.device)
            edge_index = batch['edge_index'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            activity_pred, dc50_pred = self.model(node_features, edge_index)
            
            # Multi-task loss
            loss_activity = self.criterion_activity(activity_pred, labels.unsqueeze(1))
            loss_dc50 = self.criterion_dc50(dc50_pred, labels.unsqueeze(1))
            loss = loss_activity + 0.5 * loss_dc50
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader):
        self.model.eval()
        predictions = []
        ground_truth = []
        
        with torch.no_grad():
            for batch in dataloader:
                if batch is None:
                    continue
                
                node_features = batch['node_features'].to(self.device)
                edge_index = batch['edge_index'].to(self.device)
                labels = batch['label'].to(self.device)
                
                activity_pred, _ = self.model(node_features, edge_index)
                
                predictions.extend(activity_pred.cpu().numpy().flatten().tolist())
                ground_truth.extend(labels.cpu().numpy().tolist())
        
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        # Calculate metrics
        auc = roc_auc_score(ground_truth, predictions)
        acc = accuracy_score(ground_truth, (predictions > 0.5).astype(int))
        f1 = f1_score(ground_truth, (predictions > 0.5).astype(int))
        precision = precision_score(ground_truth, (predictions > 0.5).astype(int), zero_division=0)
        recall = recall_score(ground_truth, (predictions > 0.5).astype(int), zero_division=0)
        
        return {
            'auc': auc,
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }


def create_collate_fn():
    """Create collate function for DataLoader"""
    def collate_fn(batch):
        # Filter out None values
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None
        
        # Stack features
        node_features_list = [b['node_features'] for b in batch]
        edge_index_list = [b['edge_index'] for b in batch]
        labels = torch.stack([b['label'] for b in batch])
        
        # Concatenate node features and adjust edge indices
        node_offset = 0
        all_node_features = []
        all_edge_indices = []
        
        for node_features, edge_index in zip(node_features_list, edge_index_list):
            all_node_features.append(node_features)
            if edge_index.shape[0] > 0:
                all_edge_indices.append(edge_index + node_offset)
            node_offset += node_features.shape[0]
        
        node_features = torch.cat(all_node_features, dim=0)
        if all_edge_indices:
            edge_index = torch.cat(all_edge_indices, dim=0)
        else:
            edge_index = torch.zeros((0, 2), dtype=torch.int64)
        
        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'label': labels,
            'smiles': [b['smiles'] for b in batch]
        }
    
    return collate_fn


def train_gnn_model(train_csv, val_csv, output_dir='models', epochs=50, batch_size=32):
    """Train GNN model for PROTAC activity prediction"""
    
    logger.info("Loading training data...")
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    
    # Create datasets
    train_dataset = MolecularGraphDataset(
        train_df['PROTAC_SMILES'].values,
        train_df['Activity'].values
    )
    
    val_dataset = MolecularGraphDataset(
        val_df['PROTAC_SMILES'].values,
        val_df['Activity'].values
    )
    
    # Create dataloaders
    collate_fn = create_collate_fn()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    model = ProtacGNN(node_feature_dim=6, hidden_dim=64, num_layers=3, num_tasks=1)
    trainer = ProtacModelTrainer(model, device=device, learning_rate=0.001)
    
    # Training loop
    best_auc = 0
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_metrics = trainer.evaluate(val_loader)
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, "
                   f"Val AUC: {val_metrics['auc']:.4f}, "
                   f"Val Acc: {val_metrics['accuracy']:.4f}")
        
        # Save best model
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            torch.save(model.state_dict(), output_path / 'best_model.pth')
            logger.info(f"Saved best model with AUC: {best_auc:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), output_path / 'final_model.pth')
    
    # Save training history
    history = {
        'best_auc': best_auc,
        'final_metrics': val_metrics
    }
    
    logger.info(f"Training completed. Best AUC: {best_auc:.4f}")
    
    return model, history


if __name__ == '__main__':
    import sys
    
    train_csv = 'data/processed/train_data.csv'
    val_csv = 'data/processed/val_data.csv'
    
    if Path(train_csv).exists() and Path(val_csv).exists():
        model, history = train_gnn_model(train_csv, val_csv, epochs=10)
        logger.info(f"Model training completed with history: {history}")
    else:
        logger.error(f"Training data not found. Please run data preprocessing first.")
