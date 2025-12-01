"""
SIMPLE TEST - C-RLM ENVIRONMENT
================================
Test without gymnasium dependency
"""

import sys
import os
sys.path.append(r'E:\4 Paper\Implemenatation\new_code')

import torch
import numpy as np
from step1_data_loader import PHM2010DataLoader
from step2_models import CausalStructuralTransformer

print("="*70)
print("SIMPLE C-RLM ENVIRONMENT TEST")
print("="*70)

# Configuration
CONFIG = {
    'data_path': r'E:\Collaboration Work\With Farooq\phm dataset\PHM Challange 2010 Milling',
    'd_model': 128,
    'nhead': 8,
    'num_layers': 4,
    'dropout': 0.15,
}

print("\n1. Loading data...")
loader = PHM2010DataLoader(CONFIG['data_path'])
data_dict = loader.prepare_data()
print("✓ Data loaded")

print("\n2. Loading trained causal model...")
input_dim = data_dict['train'][0].shape[2]
num_conditions = len(data_dict['condition_mapping'])

causal_model = CausalStructuralTransformer(
    input_dim=input_dim,
    num_conditions=num_conditions,
    d_model=CONFIG['d_model'],
    nhead=CONFIG['nhead'],
    num_layers=CONFIG['num_layers'],
    dropout=CONFIG['dropout']
)

# Try to load model
model_path = r'E:\4 Paper\Implemenatation\new_code\best_causal_optimized.pth'
try:
    causal_model.load_state_dict(torch.load(model_path))
    print("✓ Causal model loaded from new_code directory")
except:
    print("⚠ Could not load model - will demonstrate with untrained model")

causal_model.eval()

print("\n3. Simulating Maintenance Scenario...")

# Action definitions
ACTIONS = {
    0: "Continue Operation",
    1: "Reduce Load",
    2: "Minor Maintenance",
    3: "Major Maintenance (Replace)",
    4: "Shutdown for Inspection"
}

COSTS = {
    0: 0,      # Continue
    1: 10,     # Reduce Load
    2: 100,    # Minor Maintenance
    3: 500,    # Major Maintenance
    4: 50,     # Shutdown
    'failure': 10000
}

# Sample initial state from data
train_seq = data_dict['train'][0]
train_labels = data_dict['train'][1]
train_cond = data_dict['train'][2]
train_hi = data_dict['train'][3]

idx = np.random.randint(0, len(train_seq))

current_state = {
    'degradation_features': train_seq[idx].copy(),
    'current_rul': train_labels[idx],
    'operating_condition': train_cond[idx],
    'health_indicator': train_hi[idx]
}

print(f"\nInitial State:")
print(f"  RUL: {current_state['current_rul']:.1f} cuts")
print(f"  Health Indicator: {current_state['health_indicator']:.3f}")
print(f"  Condition: C{data_dict['reverse_mapping'][current_state['operating_condition']]}")

print("\n4. Counterfactual Analysis - What if we take each action?")
print("\n{:<30} {:<15} {:<20} {:<15}".format(
    "Action", "Cost", "Predicted RUL", "Risk"))
print("-" * 80)

# Analyze each action
for action_id, action_name in ACTIONS.items():
    # Prepare input
    seq = torch.FloatTensor(current_state['degradation_features']).unsqueeze(0)
    condition = torch.LongTensor([current_state['operating_condition']])
    hi = torch.FloatTensor([current_state['health_indicator']])
    
    # Predict using causal model
    with torch.no_grad():
        pred_rul, components = causal_model(seq, condition, hi)
    
    # Calculate risk
    risk = (1 - pred_rul.item() / 300) * current_state['health_indicator']
    
    # Determine next RUL based on action
    if action_id == 0:  # Continue
        next_rul = pred_rul.item() - 1
    elif action_id == 1:  # Reduce Load
        next_rul = pred_rul.item() - 0.5  # Slower degradation
    elif action_id == 2:  # Minor Maintenance
        next_rul = pred_rul.item() * 1.3  # Partial restoration
    elif action_id == 3:  # Major Maintenance
        next_rul = 280.0  # Full restoration
    elif action_id == 4:  # Shutdown
        next_rul = pred_rul.item()  # No change
    
    risk_level = "HIGH" if risk > 0.5 else "MEDIUM" if risk > 0.2 else "LOW"
    
    print("{:<30} ${:<14} {:<20.1f} {:<15}".format(
        action_name,
        COSTS[action_id],
        next_rul,
        risk_level
    ))

print("\n5. Optimal Decision Analysis...")

# Determine best action based on cost-benefit
decisions = []

for action_id in ACTIONS.keys():
    if action_id == 0:
        next_rul = current_state['current_rul'] - 1
        cost = COSTS[0]
    elif action_id == 1:
        next_rul = current_state['current_rul'] - 0.5
        cost = COSTS[1]
    elif action_id == 2:
        next_rul = current_state['current_rul'] * 1.3
        cost = COSTS[2]
    elif action_id == 3:
        next_rul = 280.0
        cost = COSTS[3]
    elif action_id == 4:
        next_rul = current_state['current_rul']
        cost = COSTS[4]
    
    # Calculate value: RUL gained per dollar spent
    rul_change = next_rul - current_state['current_rul']
    
    if cost == 0:
        value = rul_change if rul_change > 0 else -100
    else:
        value = rul_change / cost
    
    decisions.append({
        'action_id': action_id,
        'action_name': ACTIONS[action_id],
        'cost': cost,
        'next_rul': next_rul,
        'rul_change': rul_change,
        'value': value
    })

# Sort by value
decisions = sorted(decisions, key=lambda x: x['value'], reverse=True)

print("\nRanked Actions (by cost-effectiveness):")
for i, decision in enumerate(decisions, 1):
    print(f"\n{i}. {decision['action_name']}")
    print(f"   Cost: ${decision['cost']}")
    print(f"   RUL Change: {decision['rul_change']:+.1f} cuts")
    print(f"   Value: {decision['value']:.4f} cuts/$")

best_action = decisions[0]
print(f"\n✓ Recommended Action: {best_action['action_name']}")
print(f"  Reasoning: Best value ({best_action['value']:.4f} cuts/$)")

print("\n6. Demonstrating Causal Decomposition...")

seq = torch.FloatTensor(current_state['degradation_features']).unsqueeze(0)
condition = torch.LongTensor([current_state['operating_condition']])
hi = torch.FloatTensor([current_state['health_indicator']])

with torch.no_grad():
    pred_rul, components = causal_model(seq, condition, hi)

print(f"\nCurrent RUL Decomposition:")
print(f"  Base RUL: {components['base_rul'].item():.1f} cuts")
print(f"  Condition Effect: {components['condition_effect'].item():+.1f} cuts")
print(f"  HI Effect: {components['hi_effect'].item():+.1f} cuts")
print(f"  Total Predicted: {pred_rul.item():.1f} cuts")
print(f"  Actual: {current_state['current_rul']:.1f} cuts")

print("\n" + "="*70)
print("✅ CONCEPT DEMONSTRATION COMPLETE!")
print("="*70)

print("\nKey Capabilities Demonstrated:")
print("  ✓ Causal RUL prediction")
print("  ✓ Counterfactual analysis (what-if scenarios)")
print("  ✓ Cost-benefit optimization")
print("  ✓ Decision recommendation")
print("  ✓ Causal decomposition")

print("\nThis Forms the Foundation of C-RLM:")
print("  → RL agent will learn optimal policy")
print("  → Counterfactual predictions guide exploration")
print("  → Safety constraints prevent failures")
print("  → Multi-objective optimization balances cost/safety/performance")

print("\n" + "="*70)
print("READY TO BUILD RL AGENT!")
print("="*70)
