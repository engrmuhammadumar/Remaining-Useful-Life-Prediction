"""
Causal Transformer Models - Step 2
Two models: Baseline (accuracy) and Causal (interpretable + counterfactual)
"""

import torch
import torch.nn as nn
import numpy as np

# Check PyTorch version
print(f"PyTorch version: {torch.__version__}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class PositionalEncoding(nn.Module):
    """Add position information to sequences"""
    
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
    MODEL 1: Baseline Transformer
    Purpose: Maximum accuracy RUL prediction
    Input: Temporal sensor data
    Output: RUL prediction
    """
    
    def __init__(self, input_dim=3, d_model=128, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        
        print("\nInitializing Baseline Transformer...")
        print(f"  Input features: {input_dim}")
        print(f"  Hidden dimension: {d_model}")
        print(f"  Attention heads: {nhead}")
        print(f"  Transformer layers: {num_layers}")
        
        # Project input to d_model dimensions
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Add positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
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
        
        print(f"✓ Model created with {sum(p.numel() for p in self.parameters()):,} parameters")
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            rul: (batch,) - predicted RUL
        """
        # Project to hidden dimension
        x = self.input_projection(x)
        
        # Add position encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x)
        
        # Use last time step
        last_hidden = encoded[:, -1, :]
        
        # Predict RUL
        rul = self.fc_layers(last_hidden).squeeze(-1)
        
        return rul


class CausalStructuralTransformer(nn.Module):
    """
    MODEL 2: Causal-Structural Transformer (CST-RUL)
    Purpose: Interpretable RUL + Counterfactual analysis
    
    Innovation: Decomposes RUL into causal components:
      RUL = Base_RUL + Condition_Effect + HI_Effect
    """
    
    def __init__(self, input_dim=3, num_conditions=3, d_model=128, 
                 nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        
        print("\nInitializing Causal-Structural Transformer...")
        print(f"  Input features: {input_dim}")
        print(f"  Number of conditions: {num_conditions}")
        print(f"  Hidden dimension: {d_model}")
        
        self.num_conditions = num_conditions
        
        # Temporal feature extraction (same as baseline)
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
        
        # CAUSAL DECOMPOSITION HEADS
        
        # 1. Base RUL (from temporal patterns)
        self.base_rul_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # 2. Condition effect (causal contribution of operating condition)
        self.condition_embedding = nn.Embedding(num_conditions + 1, d_model // 4)
        self.condition_effect_network = nn.Sequential(
            nn.Linear(d_model + d_model // 4, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # 3. Health indicator effect (physics-constrained: higher wear → lower RUL)
        self.hi_effect_network = nn.Sequential(
            nn.Linear(d_model + 1, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )
        
        print(f"✓ Model created with {sum(p.numel() for p in self.parameters()):,} parameters")
        
    def forward(self, x, condition, health_indicator):
        """
        Forward pass with causal decomposition
        
        Args:
            x: (batch, seq_len, input_dim) - temporal sensor data
            condition: (batch,) - operating condition labels
            health_indicator: (batch,) - current wear state
        
        Returns:
            rul: (batch,) - predicted RUL
            components: dict with causal breakdown
        """
        # Extract temporal features
        x_proj = self.input_projection(x)
        x_pos = self.pos_encoder(x_proj)
        encoded = self.transformer_encoder(x_pos)
        temporal_features = encoded[:, -1, :]
        
        # CAUSAL DECOMPOSITION
        
        # 1. Base RUL (baseline wear progression)
        base_rul = self.base_rul_head(temporal_features).squeeze(-1)
        
        # 2. Condition causal effect
        condition_emb = self.condition_embedding(condition)
        condition_input = torch.cat([temporal_features, condition_emb], dim=-1)
        condition_effect = self.condition_effect_network(condition_input).squeeze(-1)
        
        # 3. Health indicator effect (PHYSICS CONSTRAINT: negative)
        hi_expanded = health_indicator.unsqueeze(-1)
        hi_input = torch.cat([temporal_features, hi_expanded], dim=-1)
        hi_effect = self.hi_effect_network(hi_input).squeeze(-1)
        hi_effect = -torch.abs(hi_effect)  # Ensure negative (higher wear → lower RUL)
        
        # Final RUL = sum of all components
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
        Counterfactual analysis: "What if...?"
        
        Examples:
          - What if we used Condition 4 instead of Condition 1?
          - What if we reduced wear by 20%?
        
        Args:
            x: temporal sequence
            original_condition: actual condition
            health_indicator: actual wear state
            new_condition: counterfactual condition (optional)
            new_hi: counterfactual health indicator (optional)
        
        Returns:
            Dictionary with factual, counterfactual, and delta RUL
        """
        # Factual prediction (actual scenario)
        factual_rul, factual_components = self.forward(x, original_condition, health_indicator)
        
        # Counterfactual prediction (what-if scenario)
        intervention_condition = new_condition if new_condition is not None else original_condition
        intervention_hi = new_hi if new_hi is not None else health_indicator
        
        counterfactual_rul, cf_components = self.forward(x, intervention_condition, intervention_hi)
        
        # Calculate change in RUL
        delta_rul = counterfactual_rul - factual_rul
        
        return {
            'factual_rul': factual_rul,
            'counterfactual_rul': counterfactual_rul,
            'delta_rul': delta_rul,
            'factual_components': factual_components,
            'counterfactual_components': cf_components
        }


# PyTorch Dataset wrapper
class PHM2010Dataset(torch.utils.data.Dataset):
    """Dataset wrapper for PyTorch DataLoader"""
    
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


# Test the models
if __name__ == "__main__":
    print("="*70)
    print("TESTING MODELS")
    print("="*70)
    
    # Create dummy data
    batch_size = 4
    seq_len = 20
    input_dim = 3
    
    x = torch.randn(batch_size, seq_len, input_dim)
    condition = torch.randint(0, 3, (batch_size,))
    hi = torch.rand(batch_size)
    
    # Test Baseline
    print("\n1. Testing Baseline Transformer:")
    baseline = BaselineTransformer(input_dim=input_dim)
    rul_pred = baseline(x)
    print(f"✓ Output shape: {rul_pred.shape}")
    print(f"✓ Sample predictions: {rul_pred[:3].detach().numpy()}")
    
    # Test Causal
    print("\n2. Testing Causal-Structural Transformer:")
    causal = CausalStructuralTransformer(input_dim=input_dim, num_conditions=3)
    rul_pred, components = causal(x, condition, hi)
    print(f"✓ Output shape: {rul_pred.shape}")
    print(f"✓ Sample predictions: {rul_pred[:3].detach().numpy()}")
    print(f"✓ Components available: {list(components.keys())}")
    
    # Test Counterfactual
    print("\n3. Testing Counterfactual Analysis:")
    new_condition = torch.tensor([2])  # Change to condition 2
    cf_result = causal.counterfactual_predict(
        x[0:1], condition[0:1], hi[0:1], 
        new_condition=new_condition
    )
    print(f"✓ Factual RUL: {cf_result['factual_rul'].item():.2f}")
    print(f"✓ Counterfactual RUL: {cf_result['counterfactual_rul'].item():.2f}")
    print(f"✓ Delta RUL: {cf_result['delta_rul'].item():+.2f}")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
