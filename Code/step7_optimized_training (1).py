"""
OPTIMIZED TRAINING - Fixed Convergence Issues
Better initialization and training strategy
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from step1_data_loader import PHM2010DataLoader
from step2_models import BaselineTransformer, CausalStructuralTransformer, PHM2010Dataset, device

print("="*70)
print("OPTIMIZED TRAINING - MSSP LEVEL")
print("="*70)


CONFIG = {
    'data_path': r'E:\Collaboration Work\With Farooq\phm dataset\PHM Challange 2010 Milling',
    
    # Optimized settings
    'd_model': 128,           # Back to 128 for stability
    'nhead': 8,
    'num_layers': 4,          # Back to 4 for stability
    'dropout': 0.15,          # Moderate dropout
    
    'batch_size': 32,         # Larger batch for stability
    'num_epochs': 150,
    'learning_rate': 0.001,   # Standard learning rate
    'patience': 20,
    'weight_decay': 1e-5,
    
    'sequence_length': 20,
    'warmup_epochs': 10,      # Warmup for better convergence
}

print("\nOptimized Configuration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")


def train_model_optimized(model, train_loader, val_loader, model_type='baseline'):
    """Optimized training with warmup and better convergence"""
    
    print(f"\n{'='*70}")
    print(f"TRAINING {model_type.upper()} MODEL - OPTIMIZED")
    print(f"{'='*70}")
    
    # Use Adam with warmup
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Warmup scheduler
    def lr_lambda(epoch):
        if epoch < CONFIG['warmup_epochs']:
            return (epoch + 1) / CONFIG['warmup_epochs']
        return 0.95 ** (epoch - CONFIG['warmup_epochs'])
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(CONFIG['num_epochs']):
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        
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
                mae = torch.mean(torch.abs(pred - label))
                
                val_loss += loss.item()
                val_mae += mae.item()
        
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step()
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{CONFIG['num_epochs']}] "
                  f"Train: {train_loss:.2f} | Val: {val_loss:.2f} | "
                  f"Val MAE: {val_mae:.2f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'best_{model_type}_optimized.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= CONFIG['patience']:
            print(f"\n✓ Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load(f'best_{model_type}_optimized.pth'))
    
    print(f"\n✓ Training complete! Best val loss: {best_val_loss:.4f}")
    
    return train_losses, val_losses


def evaluate_model(model, test_loader, model_type='baseline'):
    """Evaluate model"""
    
    print(f"\n{'='*70}")
    print(f"EVALUATING {model_type.upper()} MODEL")
    print(f"{'='*70}")
    
    model.eval()
    predictions = []
    actuals = []
    conditions = []
    
    with torch.no_grad():
        for batch in test_loader:
            seq = batch['sequence'].to(device)
            label = batch['label'].to(device)
            
            if model_type == 'causal':
                condition = batch['condition'].to(device)
                hi = batch['health_indicator'].to(device)
                pred, _ = model(seq, condition, hi)
                conditions.extend(condition.cpu().numpy())
            else:
                pred = model(seq)
            
            predictions.extend(pred.cpu().numpy())
            actuals.extend(label.cpu().numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    
    # Better MAPE calculation
    mask = actuals > 10
    mape = np.mean(np.abs((predictions[mask] - actuals[mask]) / actuals[mask])) * 100 if mask.sum() > 0 else 0
    
    # R-squared
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    print(f"\nOverall Performance:")
    print(f"  MAE:  {mae:.2f} cuts")
    print(f"  RMSE: {rmse:.2f} cuts")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²:   {r2:.4f}")
    
    # Per-condition analysis
    if model_type == 'causal' and len(conditions) > 0:
        conditions = np.array(conditions)
        print(f"\nPer-Condition Performance:")
        for cond_idx in np.unique(conditions):
            mask = conditions == cond_idx
            cond_mae = np.mean(np.abs(predictions[mask] - actuals[mask]))
            print(f"  Condition {cond_idx}: MAE = {cond_mae:.2f} cuts")
    
    return predictions, actuals, {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2}


def main():
    """Main optimized training pipeline"""
    
    # Load data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    loader = PHM2010DataLoader(CONFIG['data_path'])
    data_dict = loader.prepare_data(sequence_length=CONFIG['sequence_length'])
    
    train_seq, train_labels, train_cond, train_hi = data_dict['train']
    val_seq, val_labels, val_cond, val_hi = data_dict['val']
    test_seq, test_labels, test_cond, test_hi = data_dict['test']
    
    input_dim = train_seq.shape[2]
    num_conditions = len(data_dict['condition_mapping'])
    
    print(f"\n✓ Data loaded:")
    print(f"  Train: {len(train_seq)} samples")
    print(f"  Val: {len(val_seq)} samples")
    print(f"  Test: {len(test_seq)} samples")
    print(f"  Input dim: {input_dim}, Conditions: {num_conditions}")
    
    # Create datasets
    train_dataset = PHM2010Dataset(train_seq, train_labels, train_cond, train_hi)
    val_dataset = PHM2010Dataset(val_seq, val_labels, val_cond, val_hi)
    test_dataset = PHM2010Dataset(test_seq, test_labels, test_cond, test_hi)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # ========================================================================
    # Train Baseline Model
    # ========================================================================
    
    print("\n" + "="*70)
    print("BASELINE TRANSFORMER")
    print("="*70)
    
    baseline_model = BaselineTransformer(
        input_dim=input_dim,
        d_model=CONFIG['d_model'],
        nhead=CONFIG['nhead'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout']
    ).to(device)
    
    print(f"Parameters: {sum(p.numel() for p in baseline_model.parameters()):,}")
    
    train_model_optimized(baseline_model, train_loader, val_loader, model_type='baseline')
    baseline_preds, baseline_actuals, baseline_metrics = evaluate_model(
        baseline_model, test_loader, model_type='baseline'
    )
    
    # ========================================================================
    # Train Causal Model
    # ========================================================================
    
    print("\n" + "="*70)
    print("CAUSAL-STRUCTURAL TRANSFORMER")
    print("="*70)
    
    causal_model = CausalStructuralTransformer(
        input_dim=input_dim,
        num_conditions=num_conditions,
        d_model=CONFIG['d_model'],
        nhead=CONFIG['nhead'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout']
    ).to(device)
    
    print(f"Parameters: {sum(p.numel() for p in causal_model.parameters()):,}")
    
    train_model_optimized(causal_model, train_loader, val_loader, model_type='causal')
    causal_preds, causal_actuals, causal_metrics = evaluate_model(
        causal_model, test_loader, model_type='causal'
    )
    
    # ========================================================================
    # Comparison
    # ========================================================================
    
    print("\n" + "="*70)
    print("FINAL COMPARISON - OPTIMIZED TRAINING")
    print("="*70)
    
    print("\n{:<20} {:<15} {:<15}".format("Metric", "Baseline", "Causal"))
    print("-" * 50)
    print("{:<20} {:<15.2f} {:<15.2f}".format("MAE (cuts)", baseline_metrics['mae'], causal_metrics['mae']))
    print("{:<20} {:<15.2f} {:<15.2f}".format("RMSE (cuts)", baseline_metrics['rmse'], causal_metrics['rmse']))
    print("{:<20} {:<15.2f} {:<15.2f}".format("MAPE (%)", baseline_metrics['mape'], causal_metrics['mape']))
    print("{:<20} {:<15.4f} {:<15.4f}".format("R²", baseline_metrics['r2'], causal_metrics['r2']))
    
    improvement = ((baseline_metrics['mae'] - causal_metrics['mae']) / baseline_metrics['mae']) * 100
    print(f"\n✓ Causal model improvement: {improvement:.1f}%")
    
    if causal_metrics['mae'] < 3.0 and causal_metrics['r2'] > 0.95:
        print("\n✅ MSSP-LEVEL RESULTS ACHIEVED!")
    elif causal_metrics['mae'] < 5.0:
        print("\n✓ Good results, close to MSSP level")
    else:
        print("\n⚠ Results need further improvement")
    
    print("\n" + "="*70)
    print("✅ OPTIMIZED TRAINING COMPLETE!")
    print("="*70)
    
    return {
        'baseline': baseline_metrics,
        'causal': causal_metrics
    }


if __name__ == "__main__":
    results = main()
