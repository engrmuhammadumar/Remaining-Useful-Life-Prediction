"""
ENHANCED TRAINING - MSSP Level
Comprehensive experiments with better hyperparameters and validation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
from step1_data_loader import PHM2010DataLoader
from step2_models import BaselineTransformer, CausalStructuralTransformer, PHM2010Dataset, device

print("="*70)
print("ENHANCED TRAINING FOR MSSP-LEVEL RESULTS")
print("="*70)


# ============================================================================
# ENHANCED CONFIGURATION - Optimized for Better Performance
# ============================================================================

CONFIG = {
    # Data path
    'data_path': r'E:\Collaboration Work\With Farooq\phm dataset\PHM Challange 2010 Milling',
    
    # Enhanced model settings
    'd_model': 256,          # Increased from 128
    'nhead': 8,
    'num_layers': 6,         # Increased from 4
    'dropout': 0.2,          # Increased regularization
    
    # Enhanced training settings
    'batch_size': 16,        # Smaller batch for better gradients
    'num_epochs': 200,       # More training
    'learning_rate': 0.0005, # Lower learning rate
    'patience': 25,          # More patience
    'weight_decay': 1e-4,    # Stronger regularization
    
    # Data augmentation
    'sequence_length': 20,
    'use_data_augmentation': True,
    
    # Cross-validation
    'use_cross_validation': True,
    'n_folds': 5,
}

print("\nEnhanced Configuration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

def augment_sequence(seq, noise_level=0.01):
    """Add small Gaussian noise for data augmentation"""
    noise = torch.randn_like(seq) * noise_level
    return seq + noise


# ============================================================================
# ENHANCED TRAINING FUNCTION
# ============================================================================

def train_model_enhanced(model, train_loader, val_loader, num_epochs, learning_rate, 
                        model_type='baseline', patience=25, weight_decay=1e-4):
    """Enhanced training with better optimization"""
    
    print(f"\n{'='*70}")
    print(f"TRAINING {model_type.upper()} MODEL - ENHANCED")
    print(f"{'='*70}")
    
    # Better optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
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
            
            # Data augmentation
            if CONFIG['use_data_augmentation'] and np.random.rand() > 0.5:
                seq = augment_sequence(seq)
            
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
        
        scheduler.step()
        
        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping with model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, f'best_{model_type}_model_enhanced.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\n✓ Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    checkpoint = torch.load(f'best_{model_type}_model_enhanced.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\n✓ Training complete!")
    print(f"  Best val loss: {best_val_loss:.4f} at epoch {checkpoint['epoch']+1}")
    
    return train_losses, val_losses, best_val_loss


# ============================================================================
# CROSS-VALIDATION
# ============================================================================

def cross_validation_split(data_dict, n_folds=5):
    """Create cross-validation folds"""
    
    train_seq, train_labels, train_cond, train_hi = data_dict['train']
    val_seq, val_labels, val_cond, val_hi = data_dict['val']
    
    # Combine train and val for CV
    all_seq = np.concatenate([train_seq, val_seq], axis=0)
    all_labels = np.concatenate([train_labels, val_labels], axis=0)
    all_cond = np.concatenate([train_cond, val_cond], axis=0)
    all_hi = np.concatenate([train_hi, val_hi], axis=0)
    
    n_samples = len(all_seq)
    fold_size = n_samples // n_folds
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    folds = []
    for i in range(n_folds):
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < n_folds - 1 else n_samples
        
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
        
        fold = {
            'train': (all_seq[train_idx], all_labels[train_idx], 
                     all_cond[train_idx], all_hi[train_idx]),
            'val': (all_seq[val_idx], all_labels[val_idx],
                   all_cond[val_idx], all_hi[val_idx])
        }
        folds.append(fold)
    
    return folds


# ============================================================================
# COMPREHENSIVE EVALUATION
# ============================================================================

def evaluate_comprehensive(model, test_loader, model_type='baseline'):
    """Comprehensive evaluation with multiple metrics"""
    
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE EVALUATION - {model_type.upper()}")
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
    
    # Calculate comprehensive metrics
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    
    # MAPE - handle zero RUL carefully
    mask = actuals > 5  # Only calculate MAPE for RUL > 5
    mape = np.mean(np.abs((predictions[mask] - actuals[mask]) / actuals[mask])) * 100 if mask.sum() > 0 else 0
    
    # R-squared
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Max error
    max_error = np.max(np.abs(predictions - actuals))
    
    # Standard deviation of errors
    std_error = np.std(predictions - actuals)
    
    print(f"\nOverall Metrics:")
    print(f"  MAE:        {mae:.3f} cuts")
    print(f"  RMSE:       {rmse:.3f} cuts")
    print(f"  MAPE:       {mape:.2f}%")
    print(f"  R²:         {r2:.4f}")
    print(f"  Max Error:  {max_error:.3f} cuts")
    print(f"  Std Error:  {std_error:.3f} cuts")
    
    # Per-condition metrics
    if model_type == 'causal' and len(conditions) > 0:
        conditions = np.array(conditions)
        print(f"\nPer-Condition Performance:")
        for cond_idx in np.unique(conditions):
            mask = conditions == cond_idx
            cond_mae = np.mean(np.abs(predictions[mask] - actuals[mask]))
            cond_rmse = np.sqrt(np.mean((predictions[mask] - actuals[mask]) ** 2))
            print(f"  Condition {cond_idx}: MAE={cond_mae:.3f}, RMSE={cond_rmse:.3f}")
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'max_error': max_error,
        'std_error': std_error
    }
    
    return predictions, actuals, metrics


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Enhanced training pipeline"""
    
    # Load data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    loader = PHM2010DataLoader(CONFIG['data_path'])
    data_dict = loader.prepare_data(sequence_length=CONFIG['sequence_length'])
    
    test_seq, test_labels, test_cond, test_hi = data_dict['test']
    input_dim = test_seq.shape[2]
    num_conditions = len(data_dict['condition_mapping'])
    
    # Test dataset
    test_dataset = PHM2010Dataset(test_seq, test_labels, test_cond, test_hi)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # ========================================================================
    # BASELINE MODEL with Cross-Validation
    # ========================================================================
    
    print("\n" + "="*70)
    print("BASELINE TRANSFORMER - CROSS-VALIDATION")
    print("="*70)
    
    if CONFIG['use_cross_validation']:
        folds = cross_validation_split(data_dict, CONFIG['n_folds'])
        baseline_cv_scores = []
        
        for fold_idx, fold in enumerate(folds):
            print(f"\n--- Fold {fold_idx + 1}/{CONFIG['n_folds']} ---")
            
            train_dataset = PHM2010Dataset(*fold['train'])
            val_dataset = PHM2010Dataset(*fold['val'])
            
            train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                                     shuffle=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
            
            model = BaselineTransformer(
                input_dim=input_dim,
                d_model=CONFIG['d_model'],
                nhead=CONFIG['nhead'],
                num_layers=CONFIG['num_layers'],
                dropout=CONFIG['dropout']
            ).to(device)
            
            _, _, val_loss = train_model_enhanced(
                model, train_loader, val_loader,
                num_epochs=CONFIG['num_epochs'],
                learning_rate=CONFIG['learning_rate'],
                model_type='baseline',
                patience=CONFIG['patience'],
                weight_decay=CONFIG['weight_decay']
            )
            
            baseline_cv_scores.append(val_loss)
        
        print(f"\n✓ Cross-validation complete!")
        print(f"  Mean CV score: {np.mean(baseline_cv_scores):.4f} ± {np.std(baseline_cv_scores):.4f}")
    
    # Train final baseline model on full train+val
    print("\n--- Training Final Baseline Model ---")
    train_seq, train_labels, train_cond, train_hi = data_dict['train']
    val_seq, val_labels, val_cond, val_hi = data_dict['val']
    
    all_train_seq = np.concatenate([train_seq, val_seq], axis=0)
    all_train_labels = np.concatenate([train_labels, val_labels], axis=0)
    all_train_cond = np.concatenate([train_cond, val_cond], axis=0)
    all_train_hi = np.concatenate([train_hi, val_hi], axis=0)
    
    # Use 10% as validation
    val_size = len(all_train_seq) // 10
    train_dataset = PHM2010Dataset(
        all_train_seq[:-val_size], all_train_labels[:-val_size],
        all_train_cond[:-val_size], all_train_hi[:-val_size]
    )
    val_dataset = PHM2010Dataset(
        all_train_seq[-val_size:], all_train_labels[-val_size:],
        all_train_cond[-val_size:], all_train_hi[-val_size:]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                             shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    baseline_model = BaselineTransformer(
        input_dim=input_dim,
        d_model=CONFIG['d_model'],
        nhead=CONFIG['nhead'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout']
    ).to(device)
    
    train_model_enhanced(
        baseline_model, train_loader, val_loader,
        num_epochs=CONFIG['num_epochs'],
        learning_rate=CONFIG['learning_rate'],
        model_type='baseline',
        patience=CONFIG['patience'],
        weight_decay=CONFIG['weight_decay']
    )
    
    baseline_preds, baseline_actuals, baseline_metrics = evaluate_comprehensive(
        baseline_model, test_loader, model_type='baseline'
    )
    
    # ========================================================================
    # CAUSAL MODEL with Cross-Validation
    # ========================================================================
    
    print("\n" + "="*70)
    print("CAUSAL-STRUCTURAL TRANSFORMER - CROSS-VALIDATION")
    print("="*70)
    
    if CONFIG['use_cross_validation']:
        causal_cv_scores = []
        
        for fold_idx, fold in enumerate(folds):
            print(f"\n--- Fold {fold_idx + 1}/{CONFIG['n_folds']} ---")
            
            train_dataset = PHM2010Dataset(*fold['train'])
            val_dataset = PHM2010Dataset(*fold['val'])
            
            train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                                     shuffle=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
            
            model = CausalStructuralTransformer(
                input_dim=input_dim,
                num_conditions=num_conditions,
                d_model=CONFIG['d_model'],
                nhead=CONFIG['nhead'],
                num_layers=CONFIG['num_layers'],
                dropout=CONFIG['dropout']
            ).to(device)
            
            _, _, val_loss = train_model_enhanced(
                model, train_loader, val_loader,
                num_epochs=CONFIG['num_epochs'],
                learning_rate=CONFIG['learning_rate'],
                model_type='causal',
                patience=CONFIG['patience'],
                weight_decay=CONFIG['weight_decay']
            )
            
            causal_cv_scores.append(val_loss)
        
        print(f"\n✓ Cross-validation complete!")
        print(f"  Mean CV score: {np.mean(causal_cv_scores):.4f} ± {np.std(causal_cv_scores):.4f}")
    
    # Train final causal model
    print("\n--- Training Final Causal Model ---")
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                             shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    causal_model = CausalStructuralTransformer(
        input_dim=input_dim,
        num_conditions=num_conditions,
        d_model=CONFIG['d_model'],
        nhead=CONFIG['nhead'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout']
    ).to(device)
    
    train_model_enhanced(
        causal_model, train_loader, val_loader,
        num_epochs=CONFIG['num_epochs'],
        learning_rate=CONFIG['learning_rate'],
        model_type='causal',
        patience=CONFIG['patience'],
        weight_decay=CONFIG['weight_decay']
    )
    
    causal_preds, causal_actuals, causal_metrics = evaluate_comprehensive(
        causal_model, test_loader, model_type='causal'
    )
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "="*70)
    print("FINAL RESULTS - MSSP LEVEL")
    print("="*70)
    
    print("\n{:<25} {:<15} {:<15}".format("Metric", "Baseline", "Causal"))
    print("-" * 70)
    print("{:<25} {:<15.3f} {:<15.3f}".format("MAE (cuts)", baseline_metrics['mae'], causal_metrics['mae']))
    print("{:<25} {:<15.3f} {:<15.3f}".format("RMSE (cuts)", baseline_metrics['rmse'], causal_metrics['rmse']))
    print("{:<25} {:<15.2f} {:<15.2f}".format("MAPE (%)", baseline_metrics['mape'], causal_metrics['mape']))
    print("{:<25} {:<15.4f} {:<15.4f}".format("R²", baseline_metrics['r2'], causal_metrics['r2']))
    print("{:<25} {:<15.3f} {:<15.3f}".format("Max Error", baseline_metrics['max_error'], causal_metrics['max_error']))
    
    if CONFIG['use_cross_validation']:
        print("\n{:<25} {:<15} {:<15}".format("Cross-Validation", "Baseline", "Causal"))
        print("-" * 70)
        print("{:<25} {:.4f} ± {:.4f}  {:.4f} ± {:.4f}".format(
            "CV Score",
            np.mean(baseline_cv_scores), np.std(baseline_cv_scores),
            np.mean(causal_cv_scores), np.std(causal_cv_scores)
        ))
    
    # Save results
    results = {
        'config': CONFIG,
        'baseline_metrics': baseline_metrics,
        'causal_metrics': causal_metrics,
        'baseline_cv_scores': baseline_cv_scores if CONFIG['use_cross_validation'] else None,
        'causal_cv_scores': causal_cv_scores if CONFIG['use_cross_validation'] else None,
    }
    
    with open('enhanced_results.json', 'w') as f:
        json.dump(results, f, indent=4, default=str)
    
    print("\n✓ Results saved to: enhanced_results.json")
    print("\n" + "="*70)
    print("✅ ENHANCED TRAINING COMPLETE!")
    print("="*70)
    
    return results


if __name__ == "__main__":
    results = main()
