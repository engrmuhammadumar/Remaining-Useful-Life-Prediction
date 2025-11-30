"""
Training Script - Step 3
Train both Baseline and Causal models
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from step1_data_loader import PHM2010DataLoader
from step2_models import BaselineTransformer, CausalStructuralTransformer, PHM2010Dataset, device

print("="*70)
print("STEP 3: TRAINING CAUSAL TRANSFORMER MODELS")
print("="*70)


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # CHANGE THIS to your data path!
    'data_path': r'E:\Collaboration Work\With Farooq\phm dataset\PHM Challange 2010 Milling',
    
    # Model settings
    'd_model': 128,
    'nhead': 8,
    'num_layers': 4,
    'dropout': 0.1,
    
    # Training settings
    'batch_size': 32,
    'num_epochs': 50,  # Starting with 50 for faster testing
    'learning_rate': 0.001,
    'patience': 10,  # Early stopping patience
}

print("\nConfiguration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, 
                model_type='baseline', patience=10):
    """Train a model with early stopping"""
    
    print(f"\n{'='*70}")
    print(f"TRAINING {model_type.upper()} MODEL")
    print(f"{'='*70}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # PyTorch version compatibility for scheduler
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
    except TypeError:
        # Older PyTorch versions don't support verbose
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
    
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
                pred, _ = model(seq, condition, hi)
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
                    pred, _ = model(seq, condition, hi)
                else:
                    pred = model(seq)
                
                loss = criterion(pred, label)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f'best_{model_type}_model.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\n✓ Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load(f'best_{model_type}_model.pth'))
    
    print(f"\n✓ Training complete! Best val loss: {best_val_loss:.4f}")
    
    return train_losses, val_losses


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_model(model, test_loader, model_type='baseline'):
    """Evaluate model on test set"""
    
    print(f"\n{'='*70}")
    print(f"EVALUATING {model_type.upper()} MODEL")
    print(f"{'='*70}")
    
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
    
    print(f"\nResults:")
    print(f"  MAE:  {mae:.2f} cuts")
    print(f"  RMSE: {rmse:.2f} cuts")
    print(f"  MAPE: {mape:.2f}%")
    
    return predictions, actuals, {'mae': mae, 'rmse': rmse, 'mape': mape}


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Main training pipeline"""
    
    # Step 1: Load data
    print("\n" + "="*70)
    print("STEP 3.1: LOADING DATA")
    print("="*70)
    
    loader = PHM2010DataLoader(CONFIG['data_path'])
    data_dict = loader.prepare_data()
    
    train_seq, train_labels, train_cond, train_hi = data_dict['train']
    val_seq, val_labels, val_cond, val_hi = data_dict['val']
    test_seq, test_labels, test_cond, test_hi = data_dict['test']
    
    input_dim = train_seq.shape[2]
    num_conditions = len(data_dict['condition_mapping'])
    
    print(f"\n✓ Data loaded successfully!")
    print(f"  Input dimension: {input_dim}")
    print(f"  Number of conditions: {num_conditions}")
    
    # Create datasets
    train_dataset = PHM2010Dataset(train_seq, train_labels, train_cond, train_hi)
    val_dataset = PHM2010Dataset(val_seq, val_labels, val_cond, val_hi)
    test_dataset = PHM2010Dataset(test_seq, test_labels, test_cond, test_hi)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Step 2: Train Baseline Model
    print("\n" + "="*70)
    print("STEP 3.2: TRAINING BASELINE TRANSFORMER")
    print("="*70)
    
    baseline_model = BaselineTransformer(
        input_dim=input_dim,
        d_model=CONFIG['d_model'],
        nhead=CONFIG['nhead'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout']
    ).to(device)
    
    baseline_train_losses, baseline_val_losses = train_model(
        baseline_model,
        train_loader,
        val_loader,
        num_epochs=CONFIG['num_epochs'],
        learning_rate=CONFIG['learning_rate'],
        model_type='baseline',
        patience=CONFIG['patience']
    )
    
    baseline_preds, baseline_actuals, baseline_metrics = evaluate_model(
        baseline_model, test_loader, model_type='baseline'
    )
    
    # Step 3: Train Causal Model
    print("\n" + "="*70)
    print("STEP 3.3: TRAINING CAUSAL-STRUCTURAL TRANSFORMER")
    print("="*70)
    
    causal_model = CausalStructuralTransformer(
        input_dim=input_dim,
        num_conditions=num_conditions,
        d_model=CONFIG['d_model'],
        nhead=CONFIG['nhead'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout']
    ).to(device)
    
    causal_train_losses, causal_val_losses = train_model(
        causal_model,
        train_loader,
        val_loader,
        num_epochs=CONFIG['num_epochs'],
        learning_rate=CONFIG['learning_rate'],
        model_type='causal',
        patience=CONFIG['patience']
    )
    
    causal_preds, causal_actuals, causal_metrics = evaluate_model(
        causal_model, test_loader, model_type='causal'
    )
    
    # Step 4: Compare Models
    print("\n" + "="*70)
    print("STEP 3.4: MODEL COMPARISON")
    print("="*70)
    
    print("\n{:<35} {:<15} {:<15}".format("Metric", "Baseline", "Causal"))
    print("-" * 70)
    print("{:<35} {:<15.2f} {:<15.2f}".format(
        "MAE (cuts)", baseline_metrics['mae'], causal_metrics['mae']
    ))
    print("{:<35} {:<15.2f} {:<15.2f}".format(
        "RMSE (cuts)", baseline_metrics['rmse'], causal_metrics['rmse']
    ))
    print("{:<35} {:<15.2f} {:<15.2f}".format(
        "MAPE (%)", baseline_metrics['mape'], causal_metrics['mape']
    ))
    
    # Step 5: Test Counterfactual Analysis
    print("\n" + "="*70)
    print("STEP 3.5: COUNTERFACTUAL ANALYSIS DEMO")
    print("="*70)
    
    # Get one sample from test set
    test_batch = next(iter(test_loader))
    sample_seq = test_batch['sequence'][0:1].to(device)
    sample_cond = test_batch['condition'][0:1].to(device)
    sample_hi = test_batch['health_indicator'][0:1].to(device)
    sample_true_rul = test_batch['label'][0].item()
    
    # Factual prediction
    causal_model.eval()
    with torch.no_grad():
        factual_rul, components = causal_model(sample_seq, sample_cond, sample_hi)
    
    print("\nFactual Scenario (Original Condition):")
    print(f"  Condition: {data_dict['reverse_mapping'][sample_cond.item()]}")
    print(f"  Health Indicator: {sample_hi.item():.3f}")
    print(f"  True RUL: {sample_true_rul:.1f} cuts")
    print(f"  Predicted RUL: {factual_rul.item():.1f} cuts")
    print(f"\n  Causal Breakdown:")
    print(f"    Base RUL:         {components['base_rul'].item():.1f} cuts")
    print(f"    Condition Effect: {components['condition_effect'].item():+.1f} cuts")
    print(f"    HI Effect:        {components['hi_effect'].item():+.1f} cuts")
    
    # Counterfactual: Change condition
    print("\nCounterfactual: What if we used a different condition?")
    
    for new_cond_idx in range(num_conditions):
        if new_cond_idx == sample_cond.item():
            continue
        
        new_cond = torch.tensor([new_cond_idx], device=device)
        
        with torch.no_grad():
            cf_result = causal_model.counterfactual_predict(
                sample_seq, sample_cond, sample_hi,
                new_condition=new_cond
            )
        
        orig_cond_name = data_dict['reverse_mapping'][sample_cond.item()]
        new_cond_name = data_dict['reverse_mapping'][new_cond_idx]
        
        print(f"\n  Condition {orig_cond_name} → {new_cond_name}:")
        print(f"    Counterfactual RUL: {cf_result['counterfactual_rul'].item():.1f} cuts")
        print(f"    RUL Change: {cf_result['delta_rul'].item():+.1f} cuts")
        
        if cf_result['delta_rul'].item() > 0:
            print(f"    ✓ This condition would EXTEND tool life!")
        else:
            print(f"    ✗ This condition would REDUCE tool life")
    
    # Counterfactual: Reduce wear
    print("\n\nCounterfactual: What if we reduced wear by 20%?")
    reduced_hi = sample_hi * 0.8
    
    with torch.no_grad():
        cf_result = causal_model.counterfactual_predict(
            sample_seq, sample_cond, sample_hi,
            new_hi=reduced_hi
        )
    
    print(f"  Original HI: {sample_hi.item():.3f}")
    print(f"  Reduced HI:  {reduced_hi.item():.3f}")
    print(f"  Counterfactual RUL: {cf_result['counterfactual_rul'].item():.1f} cuts")
    print(f"  RUL Gain: {cf_result['delta_rul'].item():+.1f} cuts")
    print(f"  → Better maintenance could extend life by {cf_result['delta_rul'].item():.1f} cuts!")
    
    # Final Summary
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print("\nSaved Models:")
    print("  ✓ best_baseline_model.pth")
    print("  ✓ best_causal_model.pth")
    print("\nKey Results:")
    print(f"  • Baseline MAE: {baseline_metrics['mae']:.2f} cuts (high accuracy)")
    print(f"  • Causal MAE: {causal_metrics['mae']:.2f} cuts (interpretable + counterfactual)")
    print("\nNovel Capabilities:")
    print("  ✓ Causal decomposition of RUL")
    print("  ✓ What-if counterfactual analysis")
    print("  ✓ Decision support for operators")
    print("\n" + "="*70)
    
    return {
        'baseline_model': baseline_model,
        'causal_model': causal_model,
        'baseline_metrics': baseline_metrics,
        'causal_metrics': causal_metrics
    }


if __name__ == "__main__":
    results = main()
