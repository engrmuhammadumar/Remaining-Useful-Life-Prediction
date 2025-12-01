"""
ABLATION STUDIES - MSSP Level
Comprehensive experiments to validate each component
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from step1_data_loader import PHM2010DataLoader
from step2_models import CausalStructuralTransformer, PHM2010Dataset, device

print("="*70)
print("ABLATION STUDIES FOR MSSP-LEVEL VALIDATION")
print("="*70)


CONFIG = {
    'data_path': r'E:\Collaboration Work\With Farooq\phm dataset\PHM Challange 2010 Milling',
    'd_model': 256,
    'nhead': 8,
    'num_layers': 6,
    'dropout': 0.2,
    'batch_size': 16,
    'num_epochs': 100,  # Reduced for ablation
    'learning_rate': 0.0005,
    'patience': 15,
}


class CausalTransformer_NoPhysics(CausalStructuralTransformer):
    """Ablation: Remove physics constraints"""
    
    def forward(self, x, condition, health_indicator):
        x_proj = self.input_projection(x)
        x_pos = self.pos_encoder(x_proj)
        encoded = self.transformer_encoder(x_pos)
        temporal_features = encoded[:, -1, :]
        
        base_rul = self.base_rul_head(temporal_features).squeeze(-1)
        
        condition_emb = self.condition_embedding(condition)
        condition_input = torch.cat([temporal_features, condition_emb], dim=-1)
        condition_effect = self.condition_effect_network(condition_input).squeeze(-1)
        
        hi_expanded = health_indicator.unsqueeze(-1)
        hi_input = torch.cat([temporal_features, hi_expanded], dim=-1)
        hi_effect = self.hi_effect_network(hi_input).squeeze(-1)
        # NO PHYSICS CONSTRAINT - allow positive HI effect
        
        rul = base_rul + condition_effect + hi_effect
        
        components = {
            'base_rul': base_rul,
            'condition_effect': condition_effect,
            'hi_effect': hi_effect,
            'temporal_features': temporal_features
        }
        
        return rul, components


class CausalTransformer_NoCondition(nn.Module):
    """Ablation: Remove condition information"""
    
    def __init__(self, input_dim, num_conditions, d_model, nhead, num_layers, dropout):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        from step2_models import PositionalEncoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.base_rul_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        self.hi_effect_network = nn.Sequential(
            nn.Linear(d_model + 1, d_model // 4), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )
        
    def forward(self, x, condition, health_indicator):
        x_proj = self.input_projection(x)
        x_pos = self.pos_encoder(x_proj)
        encoded = self.transformer_encoder(x_pos)
        temporal_features = encoded[:, -1, :]
        
        base_rul = self.base_rul_head(temporal_features).squeeze(-1)
        
        hi_expanded = health_indicator.unsqueeze(-1)
        hi_input = torch.cat([temporal_features, hi_expanded], dim=-1)
        hi_effect = self.hi_effect_network(hi_input).squeeze(-1)
        hi_effect = -torch.abs(hi_effect)
        
        rul = base_rul + hi_effect
        
        components = {
            'base_rul': base_rul,
            'condition_effect': torch.zeros_like(base_rul),
            'hi_effect': hi_effect,
            'temporal_features': temporal_features
        }
        
        return rul, components


class CausalTransformer_NoHI(nn.Module):
    """Ablation: Remove health indicator"""
    
    def __init__(self, input_dim, num_conditions, d_model, nhead, num_layers, dropout):
        super().__init__()
        
        self.num_conditions = num_conditions
        self.input_projection = nn.Linear(input_dim, d_model)
        from step2_models import PositionalEncoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.base_rul_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        self.condition_embedding = nn.Embedding(num_conditions + 1, d_model // 4)
        self.condition_effect_network = nn.Sequential(
            nn.Linear(d_model + d_model // 4, d_model // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, x, condition, health_indicator):
        x_proj = self.input_projection(x)
        x_pos = self.pos_encoder(x_proj)
        encoded = self.transformer_encoder(x_pos)
        temporal_features = encoded[:, -1, :]
        
        base_rul = self.base_rul_head(temporal_features).squeeze(-1)
        
        condition_emb = self.condition_embedding(condition)
        condition_input = torch.cat([temporal_features, condition_emb], dim=-1)
        condition_effect = self.condition_effect_network(condition_input).squeeze(-1)
        
        rul = base_rul + condition_effect
        
        components = {
            'base_rul': base_rul,
            'condition_effect': condition_effect,
            'hi_effect': torch.zeros_like(base_rul),
            'temporal_features': temporal_features
        }
        
        return rul, components


def train_ablation_model(model, train_loader, val_loader, model_name):
    """Quick training for ablation"""
    
    print(f"\n{'='*70}")
    print(f"Training: {model_name}")
    print(f"{'='*70}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], 
                                   weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(CONFIG['num_epochs']):
        # Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            seq = batch['sequence'].to(device)
            label = batch['label'].to(device)
            condition = batch['condition'].to(device)
            hi = batch['health_indicator'].to(device)
            
            optimizer.zero_grad()
            pred, _ = model(seq, condition, hi)
            loss = criterion(pred, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                seq = batch['sequence'].to(device)
                label = batch['label'].to(device)
                condition = batch['condition'].to(device)
                hi = batch['health_indicator'].to(device)
                
                pred, _ = model(seq, condition, hi)
                loss = criterion(pred, label)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1:3d}] Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'{model_name}.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= CONFIG['patience']:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(torch.load(f'{model_name}.pth'))
    print(f"✓ Best val loss: {best_val_loss:.4f}")
    
    return best_val_loss


def evaluate_ablation(model, test_loader):
    """Evaluate ablation model"""
    
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch in test_loader:
            seq = batch['sequence'].to(device)
            label = batch['label'].to(device)
            condition = batch['condition'].to(device)
            hi = batch['health_indicator'].to(device)
            
            pred, _ = model(seq, condition, hi)
            
            predictions.extend(pred.cpu().numpy())
            actuals.extend(label.cpu().numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    
    return {'mae': mae, 'rmse': rmse}


def main():
    """Run ablation studies"""
    
    # Load data
    print("\nLoading data...")
    loader = PHM2010DataLoader(CONFIG['data_path'])
    data_dict = loader.prepare_data()
    
    train_seq, train_labels, train_cond, train_hi = data_dict['train']
    val_seq, val_labels, val_cond, val_hi = data_dict['val']
    test_seq, test_labels, test_cond, test_hi = data_dict['test']
    
    input_dim = train_seq.shape[2]
    num_conditions = len(data_dict['condition_mapping'])
    
    train_dataset = PHM2010Dataset(train_seq, train_labels, train_cond, train_hi)
    val_dataset = PHM2010Dataset(val_seq, val_labels, val_cond, val_hi)
    test_dataset = PHM2010Dataset(test_seq, test_labels, test_cond, test_hi)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    results = {}
    
    # Ablation 1: Full model (baseline)
    print("\n" + "="*70)
    print("ABLATION 1: Full Causal Model (Baseline)")
    print("="*70)
    
    model_full = CausalStructuralTransformer(
        input_dim=input_dim, num_conditions=num_conditions,
        d_model=CONFIG['d_model'], nhead=CONFIG['nhead'],
        num_layers=CONFIG['num_layers'], dropout=CONFIG['dropout']
    ).to(device)
    
    train_ablation_model(model_full, train_loader, val_loader, 'ablation_full')
    results['Full Model'] = evaluate_ablation(model_full, test_loader)
    
    # Ablation 2: No physics constraints
    print("\n" + "="*70)
    print("ABLATION 2: Without Physics Constraints")
    print("="*70)
    
    model_nophysics = CausalTransformer_NoPhysics(
        input_dim=input_dim, num_conditions=num_conditions,
        d_model=CONFIG['d_model'], nhead=CONFIG['nhead'],
        num_layers=CONFIG['num_layers'], dropout=CONFIG['dropout']
    ).to(device)
    
    train_ablation_model(model_nophysics, train_loader, val_loader, 'ablation_nophysics')
    results['No Physics'] = evaluate_ablation(model_nophysics, test_loader)
    
    # Ablation 3: No condition information
    print("\n" + "="*70)
    print("ABLATION 3: Without Condition Information")
    print("="*70)
    
    model_nocond = CausalTransformer_NoCondition(
        input_dim=input_dim, num_conditions=num_conditions,
        d_model=CONFIG['d_model'], nhead=CONFIG['nhead'],
        num_layers=CONFIG['num_layers'], dropout=CONFIG['dropout']
    ).to(device)
    
    train_ablation_model(model_nocond, train_loader, val_loader, 'ablation_nocond')
    results['No Condition'] = evaluate_ablation(model_nocond, test_loader)
    
    # Ablation 4: No health indicator
    print("\n" + "="*70)
    print("ABLATION 4: Without Health Indicator")
    print("="*70)
    
    model_nohi = CausalTransformer_NoHI(
        input_dim=input_dim, num_conditions=num_conditions,
        d_model=CONFIG['d_model'], nhead=CONFIG['nhead'],
        num_layers=CONFIG['num_layers'], dropout=CONFIG['dropout']
    ).to(device)
    
    train_ablation_model(model_nohi, train_loader, val_loader, 'ablation_nohi')
    results['No HI'] = evaluate_ablation(model_nohi, test_loader)
    
    # Print summary
    print("\n" + "="*70)
    print("ABLATION STUDY SUMMARY")
    print("="*70)
    
    df = pd.DataFrame(results).T
    df['Δ MAE'] = df['mae'] - results['Full Model']['mae']
    df['Δ RMSE'] = df['rmse'] - results['Full Model']['rmse']
    
    print("\n", df.to_string())
    
    df.to_csv('ablation_results.csv')
    print("\n✓ Results saved to: ablation_results.csv")
    
    print("\n" + "="*70)
    print("KEY FINDINGS:")
    print("="*70)
    
    for model_name, metrics in results.items():
        if model_name != 'Full Model':
            mae_diff = metrics['mae'] - results['Full Model']['mae']
            print(f"\n{model_name}:")
            print(f"  MAE increase: {mae_diff:+.3f} cuts ({mae_diff/results['Full Model']['mae']*100:+.1f}%)")
            print(f"  → {'Critical' if mae_diff > 1.0 else 'Moderate' if mae_diff > 0.5 else 'Minor'} impact")
    
    return results


if __name__ == "__main__":
    results = main()
