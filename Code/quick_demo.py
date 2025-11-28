"""
Quick Start Demo - Causal Transformer RUL Framework
This script provides a minimal working example with synthetic data for testing
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from causal_transformer_rul import (
    BaselineTransformer,
    CausalStructuralTransformer,
    PHM2010Dataset,
    device
)
from torch.utils.data import DataLoader


def generate_synthetic_data(n_samples=500, seq_len=20, n_features=3, n_conditions=3):
    """
    Generate synthetic PHM-like data for testing
    """
    print("Generating synthetic data...")
    
    sequences = []
    labels = []
    conditions = []
    health_indicators = []
    
    for i in range(n_samples):
        # Random condition
        cond = np.random.randint(0, n_conditions)
        
        # Simulate degradation
        base_degradation = np.linspace(0.1, 0.9, seq_len)
        noise = np.random.randn(seq_len, n_features) * 0.1
        
        # Condition affects degradation rate
        condition_factor = 1.0 + (cond * 0.2)
        
        # Create sequence
        seq = np.zeros((seq_len, n_features))
        for t in range(seq_len):
            wear = base_degradation[t] * condition_factor
            seq[t] = wear + noise[t]
        
        # RUL inversely related to final wear
        final_wear = seq[-1].mean()
        rul = max(0, 100 * (1 - final_wear) + np.random.randn() * 5)
        
        sequences.append(seq)
        labels.append(rul)
        conditions.append(cond)
        health_indicators.append(final_wear)
    
    sequences = np.array(sequences, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    conditions = np.array(conditions, dtype=np.int64)
    health_indicators = np.array(health_indicators, dtype=np.float32)
    
    print(f"Generated {n_samples} samples")
    print(f"  Sequences shape: {sequences.shape}")
    print(f"  Labels range: {labels.min():.1f} - {labels.max():.1f}")
    
    return sequences, labels, conditions, health_indicators


def quick_demo():
    """Run a quick demonstration"""
    
    print("="*70)
    print("CAUSAL TRANSFORMER RUL - QUICK START DEMO")
    print("="*70)
    
    # Generate synthetic data
    sequences, labels, conditions, his = generate_synthetic_data(
        n_samples=300, seq_len=20, n_features=3, n_conditions=3
    )
    
    # Split data
    n_train = 200
    n_val = 50
    
    train_seq = sequences[:n_train]
    train_labels = labels[:n_train]
    train_cond = conditions[:n_train]
    train_hi = his[:n_train]
    
    val_seq = sequences[n_train:n_train+n_val]
    val_labels = labels[n_train:n_train+n_val]
    val_cond = conditions[n_train:n_train+n_val]
    val_hi = his[n_train:n_train+n_val]
    
    test_seq = sequences[n_train+n_val:]
    test_labels = labels[n_train+n_val:]
    test_cond = conditions[n_train+n_val:]
    test_hi = his[n_train+n_val:]
    
    # Create datasets
    train_dataset = PHM2010Dataset(train_seq, train_labels, train_cond, train_hi)
    val_dataset = PHM2010Dataset(val_seq, val_labels, val_cond, val_hi)
    test_dataset = PHM2010Dataset(test_seq, test_labels, test_cond, test_hi)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print("\n" + "="*70)
    print("TRAINING CAUSAL-STRUCTURAL TRANSFORMER (DEMO)")
    print("="*70)
    
    # Initialize model
    model = CausalStructuralTransformer(
        input_dim=3,
        num_conditions=3,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        enforce_physics=True
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Quick training (just a few epochs for demo)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    print("\nTraining for 10 epochs (demo)...")
    
    for epoch in range(10):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            seq = batch['sequence'].to(device)
            label = batch['label'].to(device)
            cond = batch['condition'].to(device)
            hi = batch['health_indicator'].to(device)
            
            optimizer.zero_grad()
            pred, components = model(seq, cond, hi)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch+1}/10 - Loss: {train_loss:.4f}")
    
    print("\n" + "="*70)
    print("TESTING COUNTERFACTUAL ANALYSIS")
    print("="*70)
    
    # Get a test sample
    model.eval()
    test_batch = next(iter(test_loader))
    
    seq = test_batch['sequence'][0:1].to(device)
    orig_cond = test_batch['condition'][0:1].to(device)
    hi = test_batch['health_indicator'][0:1].to(device)
    true_rul = test_batch['label'][0].item()
    
    # Factual prediction
    with torch.no_grad():
        pred_rul, components = model(seq, orig_cond, hi)
    
    print(f"\nFactual Scenario:")
    print(f"  Condition: {orig_cond.item()}")
    print(f"  Health Indicator: {hi.item():.4f}")
    print(f"  Predicted RUL: {pred_rul.item():.2f}")
    print(f"  True RUL: {true_rul:.2f}")
    
    print(f"\n  Causal Decomposition:")
    print(f"    Base RUL: {components['base_rul'].item():.2f}")
    print(f"    Condition Effect: {components['condition_effect'].item():.2f}")
    print(f"    HI Effect: {components['hi_effect'].item():.2f}")
    
    # Counterfactual: Change condition
    print(f"\nCounterfactual Scenario 1: Change to Condition 2")
    new_cond = torch.tensor([2], device=device)
    
    with torch.no_grad():
        cf_result = model.counterfactual_predict(
            seq, orig_cond, hi,
            new_condition=new_cond
        )
    
    print(f"  Counterfactual RUL: {cf_result['counterfactual_rul'].item():.2f}")
    print(f"  RUL Change: {cf_result['delta_rul'].item():+.2f}")
    
    # Counterfactual: Reduce wear
    print(f"\nCounterfactual Scenario 2: Reduce wear by 30%")
    reduced_hi = hi * 0.7
    
    with torch.no_grad():
        cf_result2 = model.counterfactual_predict(
            seq, orig_cond, hi,
            new_hi=reduced_hi
        )
    
    print(f"  Counterfactual RUL: {cf_result2['counterfactual_rul'].item():.2f}")
    print(f"  RUL Change: {cf_result2['delta_rul'].item():+.2f}")
    
    print("\n" + "="*70)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nThis demo showed:")
    print("  ✓ Data generation and loading")
    print("  ✓ Model training (simplified)")
    print("  ✓ Causal decomposition of RUL")
    print("  ✓ Counterfactual 'What-If' analysis")
    print("\nFor full functionality with real data:")
    print("  → Run: python main_training.py")
    print("  → Or use: causal_transformer_notebook.ipynb")
    print("="*70)


if __name__ == "__main__":
    quick_demo()
