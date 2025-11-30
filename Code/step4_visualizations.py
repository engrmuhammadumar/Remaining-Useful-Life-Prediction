"""
Visualization Script - Step 4 (Final)
Create publication-quality figures for your paper
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from step1_data_loader import PHM2010DataLoader
from step2_models import BaselineTransformer, CausalStructuralTransformer, PHM2010Dataset, device

# Set style for nice plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("STEP 4: CREATING VISUALIZATIONS")
print("="*70)


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # CHANGE THIS to your data path!
    'data_path': r'E:\Collaboration Work\With Farooq\phm dataset\PHM Challange 2010 Milling',
    'd_model': 128,
    'nhead': 8,
    'num_layers': 4,
    'dropout': 0.1,
    'batch_size': 32,
}


# ============================================================================
# 1. LOAD MODELS AND DATA
# ============================================================================

print("\nLoading data and models...")

# Load data
loader = PHM2010DataLoader(CONFIG['data_path'])
data_dict = loader.prepare_data()

test_seq, test_labels, test_cond, test_hi = data_dict['test']
input_dim = test_seq.shape[2]
num_conditions = len(data_dict['condition_mapping'])

test_dataset = PHM2010Dataset(test_seq, test_labels, test_cond, test_hi)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

# Load trained models
baseline_model = BaselineTransformer(
    input_dim=input_dim,
    d_model=CONFIG['d_model'],
    nhead=CONFIG['nhead'],
    num_layers=CONFIG['num_layers'],
    dropout=CONFIG['dropout']
).to(device)
baseline_model.load_state_dict(torch.load('best_baseline_model.pth'))
baseline_model.eval()

causal_model = CausalStructuralTransformer(
    input_dim=input_dim,
    num_conditions=num_conditions,
    d_model=CONFIG['d_model'],
    nhead=CONFIG['nhead'],
    num_layers=CONFIG['num_layers'],
    dropout=CONFIG['dropout']
).to(device)
causal_model.load_state_dict(torch.load('best_causal_model.pth'))
causal_model.eval()

print("✓ Models loaded successfully!")


# ============================================================================
# 2. GET PREDICTIONS
# ============================================================================

print("\nGetting predictions...")

baseline_preds = []
causal_preds = []
actuals = []
all_conditions = []
base_ruls = []
condition_effects = []
hi_effects = []

with torch.no_grad():
    for batch in test_loader:
        seq = batch['sequence'].to(device)
        label = batch['label'].to(device)
        condition = batch['condition'].to(device)
        hi = batch['health_indicator'].to(device)
        
        # Baseline predictions
        baseline_pred = baseline_model(seq)
        baseline_preds.extend(baseline_pred.cpu().numpy())
        
        # Causal predictions
        causal_pred, components = causal_model(seq, condition, hi)
        causal_preds.extend(causal_pred.cpu().numpy())
        
        # Store components
        base_ruls.extend(components['base_rul'].cpu().numpy())
        condition_effects.extend(components['condition_effect'].cpu().numpy())
        hi_effects.extend(components['hi_effect'].cpu().numpy())
        
        actuals.extend(label.cpu().numpy())
        all_conditions.extend(condition.cpu().numpy())

baseline_preds = np.array(baseline_preds)
causal_preds = np.array(causal_preds)
actuals = np.array(actuals)
all_conditions = np.array(all_conditions)
base_ruls = np.array(base_ruls)
condition_effects = np.array(condition_effects)
hi_effects = np.array(hi_effects)

print(f"✓ Generated {len(actuals)} predictions")


# ============================================================================
# 3. FIGURE 1: PREDICTION COMPARISON
# ============================================================================

print("\nCreating Figure 1: Prediction Comparison...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Baseline predictions
axes[0].scatter(actuals, baseline_preds, alpha=0.5, s=50, c='blue')
axes[0].plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 
             'r--', lw=2, label='Perfect prediction')
axes[0].set_xlabel('True RUL (cuts)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Predicted RUL (cuts)', fontsize=12, fontweight='bold')
axes[0].set_title('Baseline Transformer', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Causal predictions
axes[1].scatter(actuals, causal_preds, alpha=0.5, s=50, c='green')
axes[1].plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 
             'r--', lw=2, label='Perfect prediction')
axes[1].set_xlabel('True RUL (cuts)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Predicted RUL (cuts)', fontsize=12, fontweight='bold')
axes[1].set_title('Causal-Structural Transformer', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

# Error comparison
baseline_errors = np.abs(baseline_preds - actuals)
causal_errors = np.abs(causal_preds - actuals)

axes[2].hist(baseline_errors, bins=30, alpha=0.5, label='Baseline', edgecolor='black', color='blue')
axes[2].hist(causal_errors, bins=30, alpha=0.5, label='Causal', edgecolor='black', color='green')
axes[2].set_xlabel('Absolute Error (cuts)', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[2].set_title('Error Distribution', fontsize=14, fontweight='bold')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('Figure1_Prediction_Comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Figure1_Prediction_Comparison.png")
plt.close()


# ============================================================================
# 4. FIGURE 2: CAUSAL DECOMPOSITION
# ============================================================================

print("\nCreating Figure 2: Causal Decomposition...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

# Select 6 samples for visualization
sample_indices = np.random.choice(len(actuals), 6, replace=False)

for idx, sample_idx in enumerate(sample_indices):
    components_data = {
        'Base\nRUL': base_ruls[sample_idx],
        'Condition\nEffect': condition_effects[sample_idx],
        'HI\nEffect': hi_effects[sample_idx],
        'Total\nRUL': causal_preds[sample_idx]
    }
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    bars = axes[idx].bar(components_data.keys(), components_data.values(), color=colors)
    
    # Add true RUL line
    axes[idx].axhline(y=actuals[sample_idx], color='black', linestyle='--', 
                      linewidth=2, label=f'True: {actuals[sample_idx]:.0f}')
    
    axes[idx].set_ylabel('RUL (cuts)', fontsize=10, fontweight='bold')
    axes[idx].set_title(f'Sample {idx+1} (Cond {data_dict["reverse_mapping"][all_conditions[sample_idx]]})',
                       fontsize=11, fontweight='bold')
    axes[idx].legend()
    axes[idx].grid(alpha=0.3, axis='y')
    axes[idx].tick_params(axis='x', rotation=0)

plt.suptitle('Causal Decomposition of RUL Predictions', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('Figure2_Causal_Decomposition.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Figure2_Causal_Decomposition.png")
plt.close()


# ============================================================================
# 5. FIGURE 3: COUNTERFACTUAL ANALYSIS
# ============================================================================

print("\nCreating Figure 3: Counterfactual Analysis...")

# Perform counterfactual analysis on multiple samples
cf_results = []

with torch.no_grad():
    for i in range(min(20, len(test_dataset))):
        sample = test_dataset[i]
        seq = sample['sequence'].unsqueeze(0).to(device)
        orig_cond = sample['condition'].unsqueeze(0).to(device)
        hi = sample['health_indicator'].unsqueeze(0).to(device)
        
        for new_cond_idx in range(num_conditions):
            if new_cond_idx == orig_cond.item():
                continue
                
            new_cond = torch.tensor([new_cond_idx], device=device)
            cf_result = causal_model.counterfactual_predict(
                seq, orig_cond, hi, new_condition=new_cond
            )
            
            cf_results.append({
                'sample': i,
                'original_condition': orig_cond.item(),
                'new_condition': new_cond_idx,
                'delta_rul': cf_result['delta_rul'].item()
            })

# Create heatmap of condition effects
condition_matrix = np.zeros((num_conditions, num_conditions))
condition_counts = np.zeros((num_conditions, num_conditions))

for result in cf_results:
    orig = result['original_condition']
    new = result['new_condition']
    condition_matrix[orig, new] += result['delta_rul']
    condition_counts[orig, new] += 1

# Average the effects
for i in range(num_conditions):
    for j in range(num_conditions):
        if condition_counts[i, j] > 0:
            condition_matrix[i, j] /= condition_counts[i, j]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Heatmap
condition_labels = [data_dict['reverse_mapping'][i] for i in range(num_conditions)]
sns.heatmap(condition_matrix, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
            xticklabels=condition_labels, yticklabels=condition_labels,
            ax=axes[0], cbar_kws={'label': 'ΔR UL (cuts)'})
axes[0].set_xlabel('New Condition', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Original Condition', fontsize=12, fontweight='bold')
axes[0].set_title('Average RUL Change by Condition Switch', fontsize=14, fontweight='bold')

# Distribution of deltas
deltas = [r['delta_rul'] for r in cf_results]
axes[1].hist(deltas, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='No change')
axes[1].set_xlabel('ΔRUL (cuts)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[1].set_title('Distribution of Counterfactual RUL Changes', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('Figure3_Counterfactual_Analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Figure3_Counterfactual_Analysis.png")
plt.close()


# ============================================================================
# 6. FIGURE 4: CONDITION EFFECTS
# ============================================================================

print("\nCreating Figure 4: Condition Effects...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Box plot of RUL by condition
condition_rul_data = {}
for cond_idx in range(num_conditions):
    mask = all_conditions == cond_idx
    condition_rul_data[data_dict['reverse_mapping'][cond_idx]] = actuals[mask]

axes[0].boxplot(condition_rul_data.values(), labels=condition_rul_data.keys())
axes[0].set_xlabel('Operating Condition', fontsize=12, fontweight='bold')
axes[0].set_ylabel('True RUL (cuts)', fontsize=12, fontweight='bold')
axes[0].set_title('RUL Distribution by Condition', fontsize=14, fontweight='bold')
axes[0].grid(alpha=0.3)

# Average condition effects
avg_condition_effects = []
for cond_idx in range(num_conditions):
    mask = all_conditions == cond_idx
    avg_effect = condition_effects[mask].mean()
    avg_condition_effects.append(avg_effect)

condition_names = [data_dict['reverse_mapping'][i] for i in range(num_conditions)]
colors_list = ['#3498db', '#2ecc71', '#e74c3c']
bars = axes[1].bar(condition_names, avg_condition_effects, color=colors_list, edgecolor='black')
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
axes[1].set_xlabel('Operating Condition', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Average Condition Effect (cuts)', fontsize=12, fontweight='bold')
axes[1].set_title('Learned Causal Effects of Conditions', fontsize=14, fontweight='bold')
axes[1].grid(alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontweight='bold')

plt.tight_layout()
plt.savefig('Figure4_Condition_Effects.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Figure4_Condition_Effects.png")
plt.close()


# ============================================================================
# 7. SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*70)
print("SUMMARY STATISTICS FOR PAPER")
print("="*70)

# Model performance
baseline_mae = np.mean(np.abs(baseline_preds - actuals))
baseline_rmse = np.sqrt(np.mean((baseline_preds - actuals) ** 2))
causal_mae = np.mean(np.abs(causal_preds - actuals))
causal_rmse = np.sqrt(np.mean((causal_preds - actuals) ** 2))

print("\nModel Performance:")
print(f"  Baseline Transformer:")
print(f"    MAE:  {baseline_mae:.2f} cuts")
print(f"    RMSE: {baseline_rmse:.2f} cuts")
print(f"\n  Causal-Structural Transformer:")
print(f"    MAE:  {causal_mae:.2f} cuts")
print(f"    RMSE: {causal_rmse:.2f} cuts")

print("\nCausal Decomposition (Average):")
print(f"  Base RUL:         {base_ruls.mean():.2f} cuts")
print(f"  Condition Effect: {condition_effects.mean():+.2f} cuts")
print(f"  HI Effect:        {hi_effects.mean():+.2f} cuts")

print("\nCounterfactual Analysis:")
print(f"  Average |ΔR UL|: {np.abs(deltas).mean():.2f} cuts")
print(f"  Max RUL gain:   {np.max(deltas):.2f} cuts")
print(f"  Max RUL loss:   {np.min(deltas):.2f} cuts")

print(f"\nBest Condition for Tool Life:")
best_condition_idx = np.argmax(avg_condition_effects)
best_condition = data_dict['reverse_mapping'][best_condition_idx]
print(f"  Condition {best_condition} (effect: {avg_condition_effects[best_condition_idx]:+.2f} cuts)")


# ============================================================================
# DONE!
# ============================================================================

print("\n" + "="*70)
print("✅ ALL VISUALIZATIONS CREATED!")
print("="*70)
print("\nGenerated Figures:")
print("  1. Figure1_Prediction_Comparison.png")
print("  2. Figure2_Causal_Decomposition.png")
print("  3. Figure3_Counterfactual_Analysis.png")
print("  4. Figure4_Condition_Effects.png")
print("\nYou can now use these figures in your paper!")
print("="*70)
