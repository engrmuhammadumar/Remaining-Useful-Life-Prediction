"""
Main Training and Evaluation Script for Causal Transformer RUL Framework
Trains both Baseline and Causal-Structural Transformers on PHM 2010 data
"""

import sys
import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import our modules
from phm2010_data_loader import PHM2010DataLoader
from causal_transformer_rul import (
    BaselineTransformer,
    CausalStructuralTransformer,
    PHM2010Dataset,
    train_model,
    evaluate_model,
    visualize_causal_decomposition,
    perform_counterfactual_analysis,
    visualize_counterfactual_results,
    plot_training_curves,
    plot_predictions_comparison,
    device
)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def main():
    """Main training and evaluation pipeline"""
    
    print("="*70)
    print(" Counterfactual RUL Estimation Using Causal Transformers")
    print(" PHM 2010 Benchmark - Milling Tool Degradation")
    print("="*70)
    
    # Configuration
    CONFIG = {
        # Data parameters
        'data_path': r'E:\Collaboration Work\With Farooq\phm dataset\PHM Challange 2010 Milling',
        'sequence_length': 20,
        'stride': 1,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        
        # Model parameters
        'd_model': 128,
        'nhead': 8,
        'num_layers': 4,
        'dropout': 0.1,
        
        # Training parameters
        'batch_size': 32,
        'num_epochs': 100,
        'learning_rate': 0.001,
        'patience': 15,
        
        # Analysis parameters
        'num_counterfactual_samples': 20,
        'num_decomposition_samples': 6
    }
    
    print("\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # ====================================================================
    # STEP 1: Load and Prepare Data
    # ====================================================================
    print("\n" + "="*70)
    print("STEP 1: Loading and Preparing Data")
    print("="*70)
    
    loader = PHM2010DataLoader(CONFIG['data_path'])
    
    try:
        data_dict = loader.prepare_data(
            sequence_length=CONFIG['sequence_length'],
            stride=CONFIG['stride'],
            train_ratio=CONFIG['train_ratio'],
            val_ratio=CONFIG['val_ratio']
        )
    except Exception as e:
        print(f"\nError loading data: {e}")
        print("\nPlease verify:")
        print(f"1. Data path exists: {CONFIG['data_path']}")
        print("2. Files are named as: c1_wear.csv, c4_wear.csv, c6_wear.csv")
        print("3. Files contain columns: cut, flute_1, flute_2, flute_3")
        return
    
    # Extract data
    train_seq, train_labels, train_cond, train_hi = data_dict['train']
    val_seq, val_labels, val_cond, val_hi = data_dict['val']
    test_seq, test_labels, test_cond, test_hi = data_dict['test']
    
    input_dim = train_seq.shape[2]  # Number of features
    num_conditions = len(data_dict['condition_mapping'])
    
    print(f"\nInput dimension: {input_dim}")
    print(f"Number of conditions: {num_conditions}")
    
    # Create datasets
    train_dataset = PHM2010Dataset(train_seq, train_labels, train_cond, train_hi)
    val_dataset = PHM2010Dataset(val_seq, val_labels, val_cond, val_hi)
    test_dataset = PHM2010Dataset(test_seq, test_labels, test_cond, test_hi)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                             shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], 
                           shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], 
                            shuffle=False)
    
    # ====================================================================
    # STEP 2: Train Baseline Transformer
    # ====================================================================
    print("\n" + "="*70)
    print("STEP 2: Training Baseline Transformer (High-Accuracy RUL Predictor)")
    print("="*70)
    
    baseline_model = BaselineTransformer(
        input_dim=input_dim,
        d_model=CONFIG['d_model'],
        nhead=CONFIG['nhead'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout']
    ).to(device)
    
    print(f"\nBaseline model parameters: {sum(p.numel() for p in baseline_model.parameters()):,}")
    
    baseline_train_losses, baseline_val_losses = train_model(
        baseline_model,
        train_loader,
        val_loader,
        num_epochs=CONFIG['num_epochs'],
        learning_rate=CONFIG['learning_rate'],
        model_type='baseline',
        patience=CONFIG['patience']
    )
    
    # Evaluate baseline
    print("\n" + "-"*70)
    print("Evaluating Baseline Transformer on Test Set")
    print("-"*70)
    
    baseline_preds, baseline_actuals, baseline_metrics = evaluate_model(
        baseline_model, test_loader, model_type='baseline'
    )
    
    # ====================================================================
    # STEP 3: Train Causal-Structural Transformer
    # ====================================================================
    print("\n" + "="*70)
    print("STEP 3: Training Causal-Structural Transformer (CST-RUL)")
    print("="*70)
    
    causal_model = CausalStructuralTransformer(
        input_dim=input_dim,
        num_conditions=num_conditions,
        d_model=CONFIG['d_model'],
        nhead=CONFIG['nhead'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout'],
        enforce_physics=True
    ).to(device)
    
    print(f"\nCausal model parameters: {sum(p.numel() for p in causal_model.parameters()):,}")
    
    causal_train_losses, causal_val_losses = train_model(
        causal_model,
        train_loader,
        val_loader,
        num_epochs=CONFIG['num_epochs'],
        learning_rate=CONFIG['learning_rate'],
        model_type='causal',
        patience=CONFIG['patience']
    )
    
    # Evaluate causal model
    print("\n" + "-"*70)
    print("Evaluating Causal-Structural Transformer on Test Set")
    print("-"*70)
    
    causal_preds, causal_actuals, causal_metrics = evaluate_model(
        causal_model, test_loader, model_type='causal'
    )
    
    # ====================================================================
    # STEP 4: Model Comparison
    # ====================================================================
    print("\n" + "="*70)
    print("STEP 4: Model Performance Comparison")
    print("="*70)
    
    comparison_df = {
        'Model': ['Baseline Transformer', 'Causal-Structural Transformer'],
        'MAE (cuts)': [baseline_metrics['mae'], causal_metrics['mae']],
        'RMSE (cuts)': [baseline_metrics['rmse'], causal_metrics['rmse']],
        'MAPE (%)': [baseline_metrics['mape'], causal_metrics['mape']]
    }
    
    import pandas as pd
    comparison_df = pd.DataFrame(comparison_df)
    print("\n", comparison_df.to_string(index=False))
    
    # ====================================================================
    # STEP 5: Causal Decomposition Analysis
    # ====================================================================
    print("\n" + "="*70)
    print("STEP 5: Causal Decomposition Analysis")
    print("="*70)
    
    base_rul, cond_eff, hi_eff, total_rul, true_rul = visualize_causal_decomposition(
        causal_model,
        test_loader,
        num_samples=CONFIG['num_decomposition_samples']
    )
    
    # Compute average contributions
    print("\nAverage Causal Contributions:")
    print(f"  Base RUL: {np.mean(base_rul):.2f} cuts")
    print(f"  Condition Effect: {np.mean(cond_eff):.2f} cuts")
    print(f"  HI Effect: {np.mean(hi_eff):.2f} cuts")
    print(f"  Total RUL: {np.mean(total_rul):.2f} cuts")
    print(f"  True RUL: {np.mean(true_rul):.2f} cuts")
    
    # ====================================================================
    # STEP 6: Counterfactual Analysis
    # ====================================================================
    print("\n" + "="*70)
    print("STEP 6: Counterfactual Analysis - What-If Scenarios")
    print("="*70)
    
    cf_results = perform_counterfactual_analysis(
        causal_model,
        test_loader,
        num_samples=CONFIG['num_counterfactual_samples']
    )
    
    # Analyze counterfactual results
    print("\nCounterfactual Insights:")
    
    # Best condition for RUL extension
    condition_changes = cf_results[cf_results['new_condition'].notna()]
    if len(condition_changes) > 0:
        avg_delta_by_condition = condition_changes.groupby('new_condition')['delta_rul'].mean()
        best_condition = avg_delta_by_condition.idxmax()
        best_gain = avg_delta_by_condition.max()
        
        reverse_map = data_dict['reverse_mapping']
        original_cond = reverse_map.get(int(best_condition), int(best_condition))
        
        print(f"\n  → Best Operating Condition: {original_cond}")
        print(f"    Average RUL gain: {best_gain:.2f} cuts")
    
    # Impact of wear reduction
    hi_reduction = cf_results[cf_results.get('intervention_type') == 'HI_reduction_20%']
    if len(hi_reduction) > 0:
        avg_hi_benefit = hi_reduction['delta_rul'].mean()
        print(f"\n  → 20% Wear Reduction Benefit:")
        print(f"    Average RUL gain: {avg_hi_benefit:.2f} cuts")
        print(f"    This demonstrates the value of improved lubrication/maintenance")
    
    # Visualize counterfactual results
    visualize_counterfactual_results(cf_results)
    
    # ====================================================================
    # STEP 7: Generate Visualizations
    # ====================================================================
    print("\n" + "="*70)
    print("STEP 7: Generating Comprehensive Visualizations")
    print("="*70)
    
    # Training curves
    plot_training_curves(
        (baseline_train_losses, baseline_val_losses),
        (causal_train_losses, causal_val_losses)
    )
    
    # Predictions comparison
    plot_predictions_comparison(baseline_preds, causal_preds, baseline_actuals)
    
    # ====================================================================
    # STEP 8: Generate Summary Report
    # ====================================================================
    print("\n" + "="*70)
    print("STEP 8: Generating Summary Report")
    print("="*70)
    
    report = f"""
    ========================================================================
    CAUSAL TRANSFORMER RUL ESTIMATION - SUMMARY REPORT
    PHM 2010 Milling Tool Degradation Benchmark
    ========================================================================
    
    DATA STATISTICS
    ----------------
    Total Samples: {len(train_dataset) + len(val_dataset) + len(test_dataset)}
    Training Samples: {len(train_dataset)}
    Validation Samples: {len(val_dataset)}
    Test Samples: {len(test_dataset)}
    Sequence Length: {CONFIG['sequence_length']} time steps
    Number of Conditions: {num_conditions}
    Input Features: {input_dim}
    
    MODEL PERFORMANCE
    -----------------
    
    Baseline Transformer:
      - MAE: {baseline_metrics['mae']:.4f} cuts
      - RMSE: {baseline_metrics['rmse']:.4f} cuts
      - MAPE: {baseline_metrics['mape']:.2f}%
      - Purpose: State-of-the-art accuracy benchmark
    
    Causal-Structural Transformer:
      - MAE: {causal_metrics['mae']:.4f} cuts
      - RMSE: {causal_metrics['rmse']:.4f} cuts
      - MAPE: {causal_metrics['mape']:.2f}%
      - Purpose: Explainable + Counterfactual RUL estimation
    
    CAUSAL INSIGHTS
    ---------------
    
    Average Causal Decomposition:
      - Base RUL: {np.mean(base_rul):.2f} cuts
      - Condition Effect: {np.mean(cond_eff):.2f} cuts
      - Health Indicator Effect: {np.mean(hi_eff):.2f} cuts
    
    Counterfactual Analysis:
      - Number of scenarios analyzed: {len(cf_results)}
      - Average RUL change across interventions: {cf_results['delta_rul'].mean():.2f} cuts
      - Maximum RUL gain observed: {cf_results['delta_rul'].max():.2f} cuts
    
    KEY CONTRIBUTIONS
    -----------------
    
    1. ✓ First application of structural causal modeling to PHM 2010
    2. ✓ Physically-interpretable RUL decomposition
    3. ✓ Counterfactual "What-If" analysis capability
    4. ✓ Decision support for process optimization
    5. ✓ Bridge between explainability and accuracy
    
    OUTPUTS GENERATED
    -----------------
    
    1. causal_decomposition.png - Visual breakdown of RUL components
    2. counterfactual_results.csv - Detailed counterfactual scenarios
    3. counterfactual_analysis.png - What-if scenario visualizations
    4. training_curves.png - Training progress for both models
    5. predictions_comparison.png - Prediction accuracy comparison
    6. best_baseline_model.pth - Trained baseline transformer
    7. best_causal_model.pth - Trained causal-structural transformer
    
    NOVEL CAPABILITIES
    ------------------
    
    ★ Question: "What if we used Condition 4 instead of Condition 1?"
      → The model can predict the RUL change
    
    ★ Question: "What if we reduced wear by 20% through better maintenance?"
      → The model quantifies the benefit
    
    ★ Question: "Which operating condition maximizes remaining tool life?"
      → The model identifies optimal conditions
    
    This transforms RUL prediction from:
      "Predicting failure" → "Evaluating decisions that prevent failure"
    
    ========================================================================
    """
    
    print(report)
    
    # Save report
    with open('/mnt/user-data/outputs/summary_report.txt', 'w') as f:
        f.write(report)
    
    print("\nSummary report saved to: summary_report.txt")
    
    # ====================================================================
    # Completion
    # ====================================================================
    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\nAll models trained, evaluated, and visualizations generated!")
    print("\nGenerated files:")
    print("  1. causal_decomposition.png")
    print("  2. counterfactual_results.csv")
    print("  3. counterfactual_analysis.png")
    print("  4. training_curves.png")
    print("  5. predictions_comparison.png")
    print("  6. summary_report.txt")
    print("  7. best_baseline_model.pth")
    print("  8. best_causal_model.pth")
    
    return {
        'baseline_model': baseline_model,
        'causal_model': causal_model,
        'data_dict': data_dict,
        'metrics': {
            'baseline': baseline_metrics,
            'causal': causal_metrics
        },
        'counterfactual_results': cf_results
    }


if __name__ == "__main__":
    results = main()
