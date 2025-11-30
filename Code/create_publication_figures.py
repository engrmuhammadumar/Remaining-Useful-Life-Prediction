"""
HIGH-IMPACT PUBLICATION FIGURES
Create beautiful, clear, publication-quality visualizations for MSSP
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader
from step1_data_loader import PHM2010DataLoader
from step2_models import BaselineTransformer, CausalStructuralTransformer, PHM2010Dataset, device

# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['figure.dpi'] = 300

print("="*70)
print("HIGH-IMPACT PUBLICATION FIGURES")
print("="*70)

# Configuration
CONFIG = {
    'data_path': r'E:\Collaboration Work\With Farooq\phm dataset\PHM Challange 2010 Milling',
    'd_model': 128,
    'nhead': 8,
    'num_layers': 4,
    'dropout': 0.15,
    'batch_size': 32,
}

# Colors for conditions
CONDITION_COLORS = {
    0: '#2E86AB',  # Blue
    1: '#A23B72',  # Purple
    2: '#F18F01',  # Orange
}

CONDITION_NAMES = {0: 'C1', 1: 'C4', 2: 'C6'}


# ============================================================================
# LOAD DATA AND MODELS
# ============================================================================

print("\nLoading data and models...")

# Load data
loader = PHM2010DataLoader(CONFIG['data_path'])
data_dict = loader.prepare_data()

train_seq, train_labels, train_cond, train_hi = data_dict['train']
val_seq, val_labels, val_cond, val_hi = data_dict['val']
test_seq, test_labels, test_cond, test_hi = data_dict['test']

input_dim = test_seq.shape[2]
num_conditions = len(data_dict['condition_mapping'])

# Create dataloaders
test_dataset = PHM2010Dataset(test_seq, test_labels, test_cond, test_hi)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

# Load models
baseline_model = BaselineTransformer(
    input_dim=input_dim, d_model=CONFIG['d_model'],
    nhead=CONFIG['nhead'], num_layers=CONFIG['num_layers'],
    dropout=CONFIG['dropout']
).to(device)
baseline_model.load_state_dict(torch.load('best_baseline_optimized.pth'))
baseline_model.eval()

causal_model = CausalStructuralTransformer(
    input_dim=input_dim, num_conditions=num_conditions,
    d_model=CONFIG['d_model'], nhead=CONFIG['nhead'],
    num_layers=CONFIG['num_layers'], dropout=CONFIG['dropout']
).to(device)
causal_model.load_state_dict(torch.load('best_causal_optimized.pth'))
causal_model.eval()

print("✓ Models loaded")

# Get predictions
print("\nGenerating predictions...")

baseline_preds = []
causal_preds = []
actuals = []
conditions = []
base_ruls = []
condition_effects = []
hi_effects = []

with torch.no_grad():
    for batch in test_loader:
        seq = batch['sequence'].to(device)
        label = batch['label'].to(device)
        condition = batch['condition'].to(device)
        hi = batch['health_indicator'].to(device)
        
        baseline_pred = baseline_model(seq)
        baseline_preds.extend(baseline_pred.cpu().numpy())
        
        causal_pred, components = causal_model(seq, condition, hi)
        causal_preds.extend(causal_pred.cpu().numpy())
        
        base_ruls.extend(components['base_rul'].cpu().numpy())
        condition_effects.extend(components['condition_effect'].cpu().numpy())
        hi_effects.extend(components['hi_effect'].cpu().numpy())
        
        actuals.extend(label.cpu().numpy())
        conditions.extend(condition.cpu().numpy())

baseline_preds = np.array(baseline_preds)
causal_preds = np.array(causal_preds)
actuals = np.array(actuals)
conditions = np.array(conditions)
base_ruls = np.array(base_ruls)
condition_effects = np.array(condition_effects)
hi_effects = np.array(hi_effects)

print(f"✓ Generated {len(actuals)} predictions")


# ============================================================================
# FIGURE 1: MODEL PERFORMANCE COMPARISON (2x2 Grid)
# ============================================================================

print("\nCreating Figure 1: Model Performance Comparison...")

fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

# (a) Baseline Predictions
ax1 = fig.add_subplot(gs[0, 0])
for cond_idx in np.unique(conditions):
    mask = conditions == cond_idx
    ax1.scatter(actuals[mask], baseline_preds[mask], 
               alpha=0.6, s=30, c=CONDITION_COLORS[cond_idx],
               label=f'Condition {data_dict["reverse_mapping"][cond_idx]}',
               edgecolors='white', linewidth=0.5)

ax1.plot([0, actuals.max()], [0, actuals.max()], 'k--', lw=2, label='Perfect prediction')
ax1.set_xlabel('True RUL (cuts)', fontweight='bold')
ax1.set_ylabel('Predicted RUL (cuts)', fontweight='bold')
ax1.set_title('(a) Baseline Transformer', fontweight='bold', loc='left')
ax1.legend(frameon=True, fancybox=True, shadow=True)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim([0, actuals.max()])
ax1.set_ylim([0, actuals.max()])

# Add metrics text
baseline_mae = np.mean(np.abs(baseline_preds - actuals))
baseline_r2 = 1 - np.sum((actuals - baseline_preds)**2) / np.sum((actuals - np.mean(actuals))**2)
ax1.text(0.05, 0.95, f'MAE = {baseline_mae:.2f} cuts\nR² = {baseline_r2:.4f}',
         transform=ax1.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# (b) Causal Predictions
ax2 = fig.add_subplot(gs[0, 1])
for cond_idx in np.unique(conditions):
    mask = conditions == cond_idx
    ax2.scatter(actuals[mask], causal_preds[mask], 
               alpha=0.6, s=30, c=CONDITION_COLORS[cond_idx],
               label=f'Condition {data_dict["reverse_mapping"][cond_idx]}',
               edgecolors='white', linewidth=0.5)

ax2.plot([0, actuals.max()], [0, actuals.max()], 'k--', lw=2, label='Perfect prediction')
ax2.set_xlabel('True RUL (cuts)', fontweight='bold')
ax2.set_ylabel('Predicted RUL (cuts)', fontweight='bold')
ax2.set_title('(b) Causal-Structural Transformer (Proposed)', fontweight='bold', loc='left')
ax2.legend(frameon=True, fancybox=True, shadow=True)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xlim([0, actuals.max()])
ax2.set_ylim([0, actuals.max()])

causal_mae = np.mean(np.abs(causal_preds - actuals))
causal_r2 = 1 - np.sum((actuals - causal_preds)**2) / np.sum((actuals - np.mean(actuals))**2)
ax2.text(0.05, 0.95, f'MAE = {causal_mae:.2f} cuts\nR² = {causal_r2:.4f}',
         transform=ax2.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# (c) Error Distribution
ax3 = fig.add_subplot(gs[1, 0])
baseline_errors = baseline_preds - actuals
causal_errors = causal_preds - actuals

ax3.hist(baseline_errors, bins=30, alpha=0.6, label='Baseline', 
         color='#3498db', edgecolor='black', linewidth=1)
ax3.hist(causal_errors, bins=30, alpha=0.6, label='Causal (Proposed)', 
         color='#2ecc71', edgecolor='black', linewidth=1)
ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero error')
ax3.set_xlabel('Prediction Error (cuts)', fontweight='bold')
ax3.set_ylabel('Frequency', fontweight='bold')
ax3.set_title('(c) Error Distribution', fontweight='bold', loc='left')
ax3.legend(frameon=True, fancybox=True, shadow=True)
ax3.grid(True, alpha=0.3, axis='y', linestyle='--')

# (d) Absolute Error Comparison
ax4 = fig.add_subplot(gs[1, 1])
baseline_abs_errors = np.abs(baseline_errors)
causal_abs_errors = np.abs(causal_errors)

positions = [1, 2]
bp = ax4.boxplot([baseline_abs_errors, causal_abs_errors],
                  positions=positions,
                  widths=0.6,
                  patch_artist=True,
                  showmeans=True,
                  meanprops=dict(marker='D', markerfacecolor='red', markersize=8))

bp['boxes'][0].set_facecolor('#3498db')
bp['boxes'][1].set_facecolor('#2ecc71')

for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp[element], color='black', linewidth=1.5)

ax4.set_ylabel('Absolute Error (cuts)', fontweight='bold')
ax4.set_title('(d) Absolute Error Comparison', fontweight='bold', loc='left')
ax4.set_xticks(positions)
ax4.set_xticklabels(['Baseline', 'Causal\n(Proposed)'], fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y', linestyle='--')

# Add improvement text
improvement = ((baseline_mae - causal_mae) / baseline_mae) * 100
ax4.text(0.5, 0.95, f'Improvement: {improvement:.1f}%',
         transform=ax4.transAxes, fontsize=11, fontweight='bold',
         verticalalignment='top', horizontalalignment='center',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

plt.savefig('Figure1_Model_Performance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Figure1_Model_Performance.png")
plt.close()


# ============================================================================
# FIGURE 2: CAUSAL DECOMPOSITION (3x2 Grid - 6 Samples)
# ============================================================================

print("\nCreating Figure 2: Causal Decomposition...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

# Select 6 diverse samples
sample_indices = []
for cond_idx in np.unique(conditions):
    cond_mask = conditions == cond_idx
    cond_samples = np.where(cond_mask)[0]
    # Select 2 samples per condition with different RUL ranges
    if len(cond_samples) >= 2:
        sample_indices.extend(np.random.choice(cond_samples, 2, replace=False))

sample_indices = sample_indices[:6]

for idx, sample_idx in enumerate(sample_indices):
    ax = axes[idx]
    
    # Components
    components_data = [
        base_ruls[sample_idx],
        condition_effects[sample_idx],
        hi_effects[sample_idx],
    ]
    
    # Cumulative sum for stacked bar
    cumsum = np.cumsum([0] + components_data)
    
    # Colors
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    labels = ['Base RUL', 'Condition\nEffect', 'HI Effect']
    
    # Draw stacked bars
    for i, (start, end, color, label) in enumerate(zip(cumsum[:-1], cumsum[1:], colors, labels)):
        height = end - start
        ax.bar(0, height, bottom=start, color=color, width=0.6, 
               label=label, edgecolor='black', linewidth=1.5)
    
    # Add total RUL line
    total_rul = causal_preds[sample_idx]
    ax.axhline(y=total_rul, color='purple', linestyle='--', linewidth=2.5, 
               label=f'Total: {total_rul:.1f}')
    
    # Add true RUL line
    true_rul = actuals[sample_idx]
    ax.axhline(y=true_rul, color='black', linestyle='-', linewidth=2.5,
               label=f'True: {true_rul:.1f}')
    
    # Styling
    ax.set_ylabel('RUL (cuts)', fontweight='bold')
    ax.set_title(f'({chr(97+idx)}) Sample {idx+1} - Cond {data_dict["reverse_mapping"][conditions[sample_idx]]}',
                fontweight='bold', loc='left')
    ax.set_xticks([])
    ax.set_xlim([-0.5, 0.5])
    ax.legend(loc='upper right', fontsize=8, frameon=True, fancybox=True)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add component values as text
    for i, val in enumerate(components_data):
        y_pos = cumsum[i] + val/2
        ax.text(0, y_pos, f'{val:.1f}', ha='center', va='center',
               fontweight='bold', fontsize=9, color='white',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

plt.suptitle('Causal Decomposition of RUL Predictions', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('Figure2_Causal_Decomposition.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Figure2_Causal_Decomposition.png")
plt.close()


# ============================================================================
# FIGURE 3: PER-CONDITION PERFORMANCE
# ============================================================================

print("\nCreating Figure 3: Per-Condition Performance...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

condition_metrics = {
    'baseline': {'mae': [], 'rmse': []},
    'causal': {'mae': [], 'rmse': []}
}

for cond_idx in sorted(np.unique(conditions)):
    mask = conditions == cond_idx
    
    baseline_mae = np.mean(np.abs(baseline_preds[mask] - actuals[mask]))
    baseline_rmse = np.sqrt(np.mean((baseline_preds[mask] - actuals[mask])**2))
    
    causal_mae = np.mean(np.abs(causal_preds[mask] - actuals[mask]))
    causal_rmse = np.sqrt(np.mean((causal_preds[mask] - actuals[mask])**2))
    
    condition_metrics['baseline']['mae'].append(baseline_mae)
    condition_metrics['baseline']['rmse'].append(baseline_rmse)
    condition_metrics['causal']['mae'].append(causal_mae)
    condition_metrics['causal']['rmse'].append(causal_rmse)

# (a) MAE per condition
ax1 = axes[0]
x = np.arange(len(np.unique(conditions)))
width = 0.35

bars1 = ax1.bar(x - width/2, condition_metrics['baseline']['mae'], width,
                label='Baseline', color='#3498db', edgecolor='black', linewidth=1.5)
bars2 = ax1.bar(x + width/2, condition_metrics['causal']['mae'], width,
                label='Causal (Proposed)', color='#2ecc71', edgecolor='black', linewidth=1.5)

ax1.set_ylabel('MAE (cuts)', fontweight='bold')
ax1.set_xlabel('Operating Condition', fontweight='bold')
ax1.set_title('(a) Mean Absolute Error by Condition', fontweight='bold', loc='left')
ax1.set_xticks(x)
ax1.set_xticklabels([f'C{data_dict["reverse_mapping"][i]}' for i in sorted(np.unique(conditions))])
ax1.legend(frameon=True, fancybox=True, shadow=True)
ax1.grid(True, alpha=0.3, axis='y', linestyle='--')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# (b) RMSE per condition
ax2 = axes[1]
bars1 = ax2.bar(x - width/2, condition_metrics['baseline']['rmse'], width,
                label='Baseline', color='#3498db', edgecolor='black', linewidth=1.5)
bars2 = ax2.bar(x + width/2, condition_metrics['causal']['rmse'], width,
                label='Causal (Proposed)', color='#2ecc71', edgecolor='black', linewidth=1.5)

ax2.set_ylabel('RMSE (cuts)', fontweight='bold')
ax2.set_xlabel('Operating Condition', fontweight='bold')
ax2.set_title('(b) Root Mean Square Error by Condition', fontweight='bold', loc='left')
ax2.set_xticks(x)
ax2.set_xticklabels([f'C{data_dict["reverse_mapping"][i]}' for i in sorted(np.unique(conditions))])
ax2.legend(frameon=True, fancybox=True, shadow=True)
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# (c) RUL distribution by condition
ax3 = axes[2]
for cond_idx in sorted(np.unique(conditions)):
    mask = conditions == cond_idx
    ax3.hist(actuals[mask], bins=20, alpha=0.5, 
            label=f'C{data_dict["reverse_mapping"][cond_idx]}',
            color=CONDITION_COLORS[cond_idx], edgecolor='black', linewidth=1)

ax3.set_xlabel('True RUL (cuts)', fontweight='bold')
ax3.set_ylabel('Frequency', fontweight='bold')
ax3.set_title('(c) RUL Distribution by Condition', fontweight='bold', loc='left')
ax3.legend(frameon=True, fancybox=True, shadow=True)
ax3.grid(True, alpha=0.3, axis='y', linestyle='--')

plt.tight_layout()
plt.savefig('Figure3_PerCondition_Performance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Figure3_PerCondition_Performance.png")
plt.close()


# ============================================================================
# FIGURE 4: COUNTERFACTUAL ANALYSIS
# ============================================================================

print("\nCreating Figure 4: Counterfactual Analysis...")

# Perform counterfactual analysis
cf_results = []

with torch.no_grad():
    for i in range(min(30, len(test_dataset))):
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
                'original_condition': orig_cond.item(),
                'new_condition': new_cond_idx,
                'delta_rul': cf_result['delta_rul'].item(),
                'factual_rul': cf_result['factual_rul'].item(),
                'counterfactual_rul': cf_result['counterfactual_rul'].item()
            })

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) Heatmap of condition switches
condition_matrix = np.zeros((num_conditions, num_conditions))
condition_counts = np.zeros((num_conditions, num_conditions))

for result in cf_results:
    orig = result['original_condition']
    new = result['new_condition']
    condition_matrix[orig, new] += result['delta_rul']
    condition_counts[orig, new] += 1

for i in range(num_conditions):
    for j in range(num_conditions):
        if condition_counts[i, j] > 0:
            condition_matrix[i, j] /= condition_counts[i, j]

ax1 = axes[0]
im = ax1.imshow(condition_matrix, cmap='RdYlGn', aspect='auto', vmin=-35, vmax=35)
ax1.set_xticks(range(num_conditions))
ax1.set_yticks(range(num_conditions))
ax1.set_xticklabels([f'C{data_dict["reverse_mapping"][i]}' for i in range(num_conditions)])
ax1.set_yticklabels([f'C{data_dict["reverse_mapping"][i]}' for i in range(num_conditions)])
ax1.set_xlabel('New Condition', fontweight='bold')
ax1.set_ylabel('Original Condition', fontweight='bold')
ax1.set_title('(a) Average RUL Change (cuts)', fontweight='bold', loc='left')

# Add text annotations
for i in range(num_conditions):
    for j in range(num_conditions):
        if i != j:
            text = ax1.text(j, i, f'{condition_matrix[i, j]:.1f}',
                          ha="center", va="center", color="black", fontweight='bold', fontsize=11)

cbar = plt.colorbar(im, ax=ax1)
cbar.set_label('ΔRUL (cuts)', rotation=270, labelpad=20, fontweight='bold')

# (b) Distribution of RUL changes
ax2 = axes[1]
deltas = [r['delta_rul'] for r in cf_results]
ax2.hist(deltas, bins=25, color='skyblue', edgecolor='black', linewidth=1.5, alpha=0.7)
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2.5, label='No change')
ax2.axvline(x=np.mean(deltas), color='green', linestyle='-', linewidth=2.5, 
           label=f'Mean: {np.mean(deltas):.1f}')
ax2.set_xlabel('ΔRUL (cuts)', fontweight='bold')
ax2.set_ylabel('Frequency', fontweight='bold')
ax2.set_title('(b) Distribution of RUL Changes', fontweight='bold', loc='left')
ax2.legend(frameon=True, fancybox=True, shadow=True)
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

# (c) Example counterfactual scenarios
ax3 = axes[2]
# Select 5 interesting examples
example_indices = np.random.choice(len(cf_results), 5, replace=False)
examples = [cf_results[i] for i in example_indices]

y_positions = np.arange(len(examples))
factual = [ex['factual_rul'] for ex in examples]
counterfactual = [ex['counterfactual_rul'] for ex in examples]

bars1 = ax3.barh(y_positions - 0.2, factual, 0.4, label='Factual',
                color='#e74c3c', edgecolor='black', linewidth=1.5)
bars2 = ax3.barh(y_positions + 0.2, counterfactual, 0.4, label='Counterfactual',
                color='#2ecc71', edgecolor='black', linewidth=1.5)

ax3.set_yticks(y_positions)
labels = [f'C{data_dict["reverse_mapping"][ex["original_condition"]]}→C{data_dict["reverse_mapping"][ex["new_condition"]]}' 
          for ex in examples]
ax3.set_yticklabels(labels)
ax3.set_xlabel('RUL (cuts)', fontweight='bold')
ax3.set_title('(c) Example Counterfactual Scenarios', fontweight='bold', loc='left')
ax3.legend(frameon=True, fancybox=True, shadow=True)
ax3.grid(True, alpha=0.3, axis='x', linestyle='--')

# Add delta annotations
for i, ex in enumerate(examples):
    delta = ex['delta_rul']
    color = 'green' if delta > 0 else 'red'
    ax3.text(max(ex['factual_rul'], ex['counterfactual_rul']) + 2, i,
            f'{delta:+.1f}', va='center', fontweight='bold', color=color, fontsize=10)

plt.tight_layout()
plt.savefig('Figure4_Counterfactual_Analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Figure4_Counterfactual_Analysis.png")
plt.close()


# ============================================================================
# FIGURE 5: LEARNED CAUSAL EFFECTS
# ============================================================================

print("\nCreating Figure 5: Learned Causal Effects...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# (a) Average component contributions
ax1 = axes[0]
components = ['Base\nRUL', 'Condition\nEffect', 'HI\nEffect']
values = [np.mean(base_ruls), np.mean(condition_effects), np.mean(hi_effects)]
colors_comp = ['#3498db', '#2ecc71', '#e74c3c']

bars = ax1.bar(components, values, color=colors_comp, edgecolor='black', linewidth=1.5, width=0.6)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax1.set_ylabel('Average Contribution (cuts)', fontweight='bold')
ax1.set_title('(a) Average Component Contributions', fontweight='bold', loc='left')
ax1.grid(True, alpha=0.3, axis='y', linestyle='--')

# Add value labels
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}', ha='center', va='bottom' if val > 0 else 'top',
            fontsize=11, fontweight='bold')

# (b) Condition effects distribution
ax2 = axes[2]
for cond_idx in sorted(np.unique(conditions)):
    mask = conditions == cond_idx
    ax2.hist(condition_effects[mask], bins=20, alpha=0.6,
            label=f'C{data_dict["reverse_mapping"][cond_idx]}',
            color=CONDITION_COLORS[cond_idx], edgecolor='black', linewidth=1)

ax2.axvline(x=0, color='black', linestyle='--', linewidth=2)
ax2.set_xlabel('Condition Effect (cuts)', fontweight='bold')
ax2.set_ylabel('Frequency', fontweight='bold')
ax2.set_title('(b) Distribution of Condition Effects', fontweight='bold', loc='left')
ax2.legend(frameon=True, fancybox=True, shadow=True)
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

# (c) HI effect vs Health Indicator
ax3 = axes[1]
for cond_idx in sorted(np.unique(conditions)):
    mask = conditions == cond_idx
    hi_vals = test_hi[mask]
    hi_eff_vals = hi_effects[mask]
    ax3.scatter(hi_vals, hi_eff_vals, alpha=0.6, s=40,
               label=f'C{data_dict["reverse_mapping"][cond_idx]}',
               color=CONDITION_COLORS[cond_idx], edgecolors='white', linewidth=0.5)

ax3.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax3.set_xlabel('Health Indicator (Normalized)', fontweight='bold')
ax3.set_ylabel('HI Effect on RUL (cuts)', fontweight='bold')
ax3.set_title('(c) Health Indicator Effect (Physics-Constrained)', fontweight='bold', loc='left')
ax3.legend(frameon=True, fancybox=True, shadow=True)
ax3.grid(True, alpha=0.3, linestyle='--')

# Add annotation about physics constraint
ax3.text(0.5, 0.95, 'Physics Constraint:\nHigher HI → Lower RUL',
        transform=ax3.transAxes, fontsize=10, fontweight='bold',
        verticalalignment='top', horizontalalignment='center',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

plt.tight_layout()
plt.savefig('Figure5_Causal_Effects.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Figure5_Causal_Effects.png")
plt.close()


# ============================================================================
# FIGURE 6: RESIDUAL ANALYSIS
# ============================================================================

print("\nCreating Figure 6: Residual Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) Residuals vs Predicted (Baseline)
ax1 = axes[0, 0]
ax1.scatter(baseline_preds, baseline_preds - actuals, alpha=0.5, s=30, color='#3498db')
ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax1.set_xlabel('Predicted RUL (cuts)', fontweight='bold')
ax1.set_ylabel('Residuals (cuts)', fontweight='bold')
ax1.set_title('(a) Baseline: Residual vs Predicted', fontweight='bold', loc='left')
ax1.grid(True, alpha=0.3, linestyle='--')

# (b) Residuals vs Predicted (Causal)
ax2 = axes[0, 1]
ax2.scatter(causal_preds, causal_preds - actuals, alpha=0.5, s=30, color='#2ecc71')
ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('Predicted RUL (cuts)', fontweight='bold')
ax2.set_ylabel('Residuals (cuts)', fontweight='bold')
ax2.set_title('(b) Causal: Residual vs Predicted', fontweight='bold', loc='left')
ax2.grid(True, alpha=0.3, linestyle='--')

# (c) Q-Q Plot (Baseline)
ax3 = axes[1, 0]
from scipy import stats
stats.probplot(baseline_preds - actuals, dist="norm", plot=ax3)
ax3.set_title('(c) Baseline: Q-Q Plot', fontweight='bold', loc='left')
ax3.grid(True, alpha=0.3, linestyle='--')

# (d) Q-Q Plot (Causal)
ax4 = axes[1, 1]
stats.probplot(causal_preds - actuals, dist="norm", plot=ax4)
ax4.set_title('(d) Causal: Q-Q Plot', fontweight='bold', loc='left')
ax4.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('Figure6_Residual_Analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Figure6_Residual_Analysis.png")
plt.close()


# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("✅ ALL HIGH-IMPACT FIGURES CREATED!")
print("="*70)
print("\nGenerated Figures:")
print("  1. Figure1_Model_Performance.png      - 2x2 grid showing model comparison")
print("  2. Figure2_Causal_Decomposition.png   - 6 samples with component breakdown")
print("  3. Figure3_PerCondition_Performance.png - Performance by operating condition")
print("  4. Figure4_Counterfactual_Analysis.png  - What-if scenario analysis")
print("  5. Figure5_Causal_Effects.png          - Learned causal relationships")
print("  6. Figure6_Residual_Analysis.png       - Statistical validation")
print("\nAll figures are publication-ready at 300 DPI!")
print("="*70)
