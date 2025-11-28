"""
Counterfactual Remaining Useful Life Estimation Using Causal Transformers
A Structural Intervention Framework for Milling Tool Degradation (PHM 2010 Benchmark)

Author: Muhammad Umar
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class PHM2010Dataset(Dataset):
    """Dataset class for PHM 2010 milling data with RUL labels"""
    
    def __init__(self, sequences, labels, conditions, health_indicators):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)
        self.conditions = torch.LongTensor(conditions)
        self.health_indicators = torch.FloatTensor(health_indicators)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequence': self.sequences[idx],
            'label': self.labels[idx],
            'condition': self.conditions[idx],
            'health_indicator': self.health_indicators[idx]
        }


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class BaselineTransformer(nn.Module):
    """
    Baseline Transformer for high-accuracy RUL prediction.
    Learns latent temporal wear dynamics without explicit causal structure.
    """
    
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # RUL prediction head
        self.fc_layers = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x)
        
        # Use last time step for prediction
        last_hidden = encoded[:, -1, :]
        
        # RUL prediction
        rul = self.fc_layers(last_hidden).squeeze(-1)
        
        return rul


class CausalStructuralTransformer(nn.Module):
    """
    Causal-Structural Transformer (CST-RUL) for interpretable and counterfactual RUL estimation.
    
    Decomposes RUL into:
    - Base wear progression (from temporal patterns)
    - Causal contributions from operating conditions
    - Health indicator (wear state) contribution
    
    Predicted RUL = Base_RUL + Σ(Condition_Effects) + HI_Effect
    """
    
    def __init__(self, input_dim, num_conditions=3, d_model=128, nhead=8, 
                 num_layers=4, dropout=0.1, enforce_physics=True):
        super().__init__()
        
        self.num_conditions = num_conditions
        self.enforce_physics = enforce_physics
        
        # Temporal feature extraction (similar to baseline)
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Base RUL prediction (baseline wear progression)
        self.base_rul_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # Causal condition embeddings
        self.condition_embedding = nn.Embedding(num_conditions + 1, d_model // 4)  # +1 for padding
        
        # Causal effect coefficients for each condition
        # These represent the causal contribution of each condition to RUL
        self.condition_effect_network = nn.Sequential(
            nn.Linear(d_model + d_model // 4, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # Health indicator (wear state) effect network
        # Physics: Higher wear should decrease RUL (negative contribution)
        self.hi_effect_network = nn.Sequential(
            nn.Linear(d_model + 1, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )
        
    def forward(self, x, condition, health_indicator):
        """
        Args:
            x: (batch, seq_len, input_dim) - temporal sensor data
            condition: (batch,) - operating condition labels
            health_indicator: (batch,) - current wear state (e.g., avg flute wear)
        
        Returns:
            rul: (batch,) - predicted RUL
            components: dict with causal decomposition
        """
        batch_size = x.size(0)
        
        # Extract temporal features
        x_proj = self.input_projection(x)
        x_pos = self.pos_encoder(x_proj)
        encoded = self.transformer_encoder(x_pos)
        temporal_features = encoded[:, -1, :]  # (batch, d_model)
        
        # 1. Base RUL (baseline wear progression)
        base_rul = self.base_rul_head(temporal_features).squeeze(-1)  # (batch,)
        
        # 2. Condition causal effect
        condition_emb = self.condition_embedding(condition)  # (batch, d_model//4)
        condition_input = torch.cat([temporal_features, condition_emb], dim=-1)
        condition_effect = self.condition_effect_network(condition_input).squeeze(-1)  # (batch,)
        
        # 3. Health indicator effect
        hi_expanded = health_indicator.unsqueeze(-1)  # (batch, 1)
        hi_input = torch.cat([temporal_features, hi_expanded], dim=-1)
        hi_effect = self.hi_effect_network(hi_input).squeeze(-1)  # (batch,)
        
        # Physics constraint: Higher wear (HI) should reduce RUL
        if self.enforce_physics:
            hi_effect = -torch.abs(hi_effect)  # Ensure negative contribution
        
        # Final RUL decomposition
        rul = base_rul + condition_effect + hi_effect
        
        # Store components for interpretability
        components = {
            'base_rul': base_rul,
            'condition_effect': condition_effect,
            'hi_effect': hi_effect,
            'temporal_features': temporal_features
        }
        
        return rul, components
    
    def counterfactual_predict(self, x, original_condition, health_indicator, 
                              new_condition=None, new_hi=None):
        """
        Perform counterfactual RUL estimation via structural intervention.
        
        do(condition = new_condition) or do(HI = new_hi)
        
        Args:
            x: temporal sequence
            original_condition: actual operating condition
            health_indicator: actual wear state
            new_condition: counterfactual condition (optional)
            new_hi: counterfactual health indicator (optional)
        
        Returns:
            factual_rul: RUL under actual conditions
            counterfactual_rul: RUL under intervention
            delta_rul: change in RUL due to intervention
        """
        # Get factual prediction
        factual_rul, factual_components = self.forward(x, original_condition, health_indicator)
        
        # Perform intervention
        intervention_condition = new_condition if new_condition is not None else original_condition
        intervention_hi = new_hi if new_hi is not None else health_indicator
        
        counterfactual_rul, cf_components = self.forward(x, intervention_condition, intervention_hi)
        
        delta_rul = counterfactual_rul - factual_rul
        
        return {
            'factual_rul': factual_rul,
            'counterfactual_rul': counterfactual_rul,
            'delta_rul': delta_rul,
            'factual_components': factual_components,
            'counterfactual_components': cf_components
        }


class PhysicsInformedLoss(nn.Module):
    """
    Loss function with physics-informed regularization
    """
    
    def __init__(self, lambda_physics=0.1):
        super().__init__()
        self.lambda_physics = lambda_physics
        self.mse_loss = nn.MSELoss()
        
    def forward(self, pred_rul, true_rul, components=None):
        # Standard MSE loss
        mse = self.mse_loss(pred_rul, true_rul)
        
        if components is None:
            return mse
        
        # Physics regularization: HI effect should be negative
        # (higher wear should reduce RUL)
        hi_effect = components['hi_effect']
        physics_violation = F.relu(hi_effect).mean()  # Penalize positive HI effects
        
        total_loss = mse + self.lambda_physics * physics_violation
        
        return total_loss, mse, physics_violation


def train_model(model, train_loader, val_loader, num_epochs=100, 
                learning_rate=0.001, model_type='baseline', patience=15):
    """Train the model with early stopping"""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    if model_type == 'causal':
        criterion = PhysicsInformedLoss(lambda_physics=0.1)
    else:
        criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            seq = batch['sequence'].to(device)
            label = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            if model_type == 'causal':
                condition = batch['condition'].to(device)
                hi = batch['health_indicator'].to(device)
                pred, components = model(seq, condition, hi)
                loss, mse, phys_loss = criterion(pred, label, components)
            else:
                pred = model(seq)
                loss = criterion(pred, label)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                seq = batch['sequence'].to(device)
                label = batch['label'].to(device)
                
                if model_type == 'causal':
                    condition = batch['condition'].to(device)
                    hi = batch['health_indicator'].to(device)
                    pred, components = model(seq, condition, hi)
                    loss, mse, _ = criterion(pred, label, components)
                else:
                    pred = model(seq)
                    loss = criterion(pred, label)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'/home/claude/best_{model_type}_model.pth')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load(f'/home/claude/best_{model_type}_model.pth'))
    
    return train_losses, val_losses


def evaluate_model(model, test_loader, model_type='baseline'):
    """Evaluate model performance"""
    
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch in test_loader:
            seq = batch['sequence'].to(device)
            label = batch['label'].to(device)
            
            if model_type == 'causal':
                condition = batch['condition'].to(device)
                hi = batch['health_indicator'].to(device)
                pred, _ = model(seq, condition, hi)
            else:
                pred = model(seq)
            
            predictions.extend(pred.cpu().numpy())
            actuals.extend(label.cpu().numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    mape = np.mean(np.abs((predictions - actuals) / (actuals + 1e-8))) * 100
    
    print(f"\n{model_type.upper()} Model Performance:")
    print(f"MAE: {mae:.4f} cuts")
    print(f"RMSE: {rmse:.4f} cuts")
    print(f"MAPE: {mape:.2f}%")
    
    return predictions, actuals, {'mae': mae, 'rmse': rmse, 'mape': mape}


def visualize_causal_decomposition(model, test_loader, num_samples=5):
    """Visualize the causal decomposition of RUL predictions"""
    
    model.eval()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    samples_collected = 0
    all_base_rul = []
    all_condition_effect = []
    all_hi_effect = []
    all_total_rul = []
    all_true_rul = []
    
    with torch.no_grad():
        for batch in test_loader:
            if samples_collected >= num_samples:
                break
                
            seq = batch['sequence'].to(device)
            condition = batch['condition'].to(device)
            hi = batch['health_indicator'].to(device)
            true_rul = batch['label'].numpy()
            
            pred_rul, components = model(seq, condition, hi)
            
            for i in range(min(seq.size(0), num_samples - samples_collected)):
                base = components['base_rul'][i].cpu().item()
                cond_eff = components['condition_effect'][i].cpu().item()
                hi_eff = components['hi_effect'][i].cpu().item()
                total = pred_rul[i].cpu().item()
                
                all_base_rul.append(base)
                all_condition_effect.append(cond_eff)
                all_hi_effect.append(hi_eff)
                all_total_rul.append(total)
                all_true_rul.append(true_rul[i])
                
                # Plot individual decomposition
                if samples_collected < 6:
                    components_data = {
                        'Base RUL': base,
                        'Condition\nEffect': cond_eff,
                        'HI Effect': hi_eff,
                        'Total RUL': total
                    }
                    
                    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
                    bars = axes[samples_collected].bar(
                        components_data.keys(), 
                        components_data.values(),
                        color=colors
                    )
                    axes[samples_collected].axhline(y=true_rul[i], color='black', 
                                                   linestyle='--', label=f'True: {true_rul[i]:.1f}')
                    axes[samples_collected].set_ylabel('RUL (cuts)')
                    axes[samples_collected].set_title(f'Sample {samples_collected + 1}\n'
                                                     f'Condition: {condition[i].item()}')
                    axes[samples_collected].legend()
                    axes[samples_collected].grid(alpha=0.3)
                
                samples_collected += 1
                
                if samples_collected >= num_samples:
                    break
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/causal_decomposition.png', dpi=300, bbox_inches='tight')
    print("Causal decomposition visualization saved!")
    
    return all_base_rul, all_condition_effect, all_hi_effect, all_total_rul, all_true_rul


def perform_counterfactual_analysis(model, test_loader, num_samples=10):
    """Perform counterfactual analysis: What-if scenarios"""
    
    model.eval()
    
    results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= num_samples:
                break
            
            seq = batch['sequence'].to(device)
            original_condition = batch['condition'].to(device)
            hi = batch['health_indicator'].to(device)
            true_rul = batch['label']
            
            # Take first sample in batch
            seq_single = seq[0:1]
            orig_cond = original_condition[0:1]
            hi_single = hi[0:1]
            
            # Scenario 1: What if we changed the operating condition?
            available_conditions = [0, 1, 2]  # Assuming conditions 1, 4, 6 mapped to 0, 1, 2
            
            for new_cond_idx in available_conditions:
                new_cond = torch.tensor([new_cond_idx], device=device)
                
                cf_result = model.counterfactual_predict(
                    seq_single, orig_cond, hi_single,
                    new_condition=new_cond
                )
                
                results.append({
                    'sample_id': batch_idx,
                    'original_condition': orig_cond.item(),
                    'new_condition': new_cond_idx,
                    'original_hi': hi_single.item(),
                    'factual_rul': cf_result['factual_rul'].item(),
                    'counterfactual_rul': cf_result['counterfactual_rul'].item(),
                    'delta_rul': cf_result['delta_rul'].item(),
                    'true_rul': true_rul[0].item()
                })
            
            # Scenario 2: What if we reduced wear by 20%?
            reduced_hi = hi_single * 0.8
            cf_result = model.counterfactual_predict(
                seq_single, orig_cond, hi_single,
                new_hi=reduced_hi
            )
            
            results.append({
                'sample_id': batch_idx,
                'original_condition': orig_cond.item(),
                'new_condition': orig_cond.item(),
                'original_hi': hi_single.item(),
                'counterfactual_hi': reduced_hi.item(),
                'factual_rul': cf_result['factual_rul'].item(),
                'counterfactual_rul': cf_result['counterfactual_rul'].item(),
                'delta_rul': cf_result['delta_rul'].item(),
                'true_rul': true_rul[0].item(),
                'intervention_type': 'HI_reduction_20%'
            })
    
    cf_df = pd.DataFrame(results)
    cf_df.to_csv('/mnt/user-data/outputs/counterfactual_results.csv', index=False)
    
    print("\nCounterfactual Analysis Results:")
    print(cf_df.head(10))
    
    return cf_df


def visualize_counterfactual_results(cf_df):
    """Visualize counterfactual analysis results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Effect of condition changes on RUL
    condition_changes = cf_df[cf_df['new_condition'].notna()]
    if len(condition_changes) > 0:
        pivot_data = condition_changes.pivot_table(
            values='delta_rul',
            index='original_condition',
            columns='new_condition',
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn', 
                    center=0, ax=axes[0, 0], cbar_kws={'label': 'ΔR UL (cuts)'})
        axes[0, 0].set_title('Average RUL Change by Condition Switch', fontsize=12, weight='bold')
        axes[0, 0].set_xlabel('New Condition')
        axes[0, 0].set_ylabel('Original Condition')
    
    # Plot 2: Distribution of RUL changes
    axes[0, 1].hist(cf_df['delta_rul'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=0, color='red', linestyle='--', label='No change')
    axes[0, 1].set_xlabel('ΔRUL (cuts)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Counterfactual RUL Changes', fontsize=12, weight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Plot 3: HI reduction impact
    hi_reduction = cf_df[cf_df.get('intervention_type') == 'HI_reduction_20%']
    if len(hi_reduction) > 0:
        axes[1, 0].scatter(hi_reduction['original_hi'], hi_reduction['delta_rul'], 
                          alpha=0.6, s=100)
        axes[1, 0].set_xlabel('Original Health Indicator (Wear)')
        axes[1, 0].set_ylabel('ΔRUL from 20% Wear Reduction (cuts)')
        axes[1, 0].set_title('Impact of Wear Reduction on RUL', fontsize=12, weight='bold')
        axes[1, 0].grid(alpha=0.3)
    
    # Plot 4: Factual vs Counterfactual RUL comparison
    sample_data = cf_df[cf_df['sample_id'] < 5]
    x = np.arange(len(sample_data))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, sample_data['factual_rul'], width, 
                   label='Factual RUL', alpha=0.8)
    axes[1, 1].bar(x + width/2, sample_data['counterfactual_rul'], width,
                   label='Counterfactual RUL', alpha=0.8)
    axes[1, 1].set_xlabel('Sample ID')
    axes[1, 1].set_ylabel('RUL (cuts)')
    axes[1, 1].set_title('Factual vs Counterfactual RUL Comparison', fontsize=12, weight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/counterfactual_analysis.png', dpi=300, bbox_inches='tight')
    print("Counterfactual analysis visualization saved!")


def plot_training_curves(baseline_losses, causal_losses):
    """Plot training curves for both models"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Baseline model
    axes[0].plot(baseline_losses[0], label='Train Loss', linewidth=2)
    axes[0].plot(baseline_losses[1], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Baseline Transformer Training', fontsize=12, weight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Causal model
    axes[1].plot(causal_losses[0], label='Train Loss', linewidth=2)
    axes[1].plot(causal_losses[1], label='Val Loss', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Causal-Structural Transformer Training', fontsize=12, weight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/training_curves.png', dpi=300, bbox_inches='tight')
    print("Training curves saved!")


def plot_predictions_comparison(baseline_preds, causal_preds, actuals):
    """Compare predictions from both models"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Baseline predictions
    axes[0].scatter(actuals, baseline_preds, alpha=0.5, s=50)
    axes[0].plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 
                 'r--', lw=2, label='Perfect prediction')
    axes[0].set_xlabel('True RUL (cuts)')
    axes[0].set_ylabel('Predicted RUL (cuts)')
    axes[0].set_title('Baseline Transformer', fontsize=12, weight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Causal predictions
    axes[1].scatter(actuals, causal_preds, alpha=0.5, s=50, color='green')
    axes[1].plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 
                 'r--', lw=2, label='Perfect prediction')
    axes[1].set_xlabel('True RUL (cuts)')
    axes[1].set_ylabel('Predicted RUL (cuts)')
    axes[1].set_title('Causal-Structural Transformer', fontsize=12, weight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # Error comparison
    baseline_errors = np.abs(baseline_preds - actuals)
    causal_errors = np.abs(causal_preds - actuals)
    
    axes[2].hist(baseline_errors, bins=30, alpha=0.5, label='Baseline', edgecolor='black')
    axes[2].hist(causal_errors, bins=30, alpha=0.5, label='Causal', edgecolor='black')
    axes[2].set_xlabel('Absolute Error (cuts)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Error Distribution Comparison', fontsize=12, weight='bold')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/predictions_comparison.png', dpi=300, bbox_inches='tight')
    print("Predictions comparison saved!")


print("Causal Transformer RUL Framework loaded successfully!")
print("Ready to load PHM 2010 data and train models.")
