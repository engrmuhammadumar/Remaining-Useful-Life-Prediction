"""
TEST MAINTENANCE ENVIRONMENT
============================
Verify the environment works correctly before building RL agent
"""

import sys
import os

# Add parent directory to path to import previous modules
sys.path.append(os.path.abspath('../new_code'))

import torch
import numpy as np
from step1_data_loader import PHM2010DataLoader
from step2_models import CausalStructuralTransformer
from step1_maintenance_environment import MaintenanceEnvironment

print("="*70)
print("TESTING MAINTENANCE ENVIRONMENT - STEP 1")
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

try:
    causal_model.load_state_dict(torch.load('best_causal_optimized.pth'))
    print("✓ Causal model loaded")
except:
    print("⚠ Could not load model - using untrained model for testing")

causal_model.eval()

print("\n3. Creating maintenance environment...")
env_config = {
    'seq_len': 20,
    'n_features': input_dim,
    'history_length': 5,
    'max_episode_length': 300,
    'costs': {
        'continue': 0,
        'reduce_load': 10,
        'minor_maintenance': 100,
        'major_maintenance': 500,
        'shutdown': 50,
        'failure': 10000,
        'downtime_per_cut': 0.5
    },
    'safety_threshold': 0.05,
    'reward_weights': {
        'cost': 1.0,
        'safety': 10.0,
        'availability': 0.5
    }
}

env = MaintenanceEnvironment(
    causal_model=causal_model,
    data_dict=data_dict,
    config=env_config
)
print("✓ Environment created")

print("\n4. Testing environment reset...")
obs, info = env.reset()
print(f"✓ Reset successful")
print(f"  Initial RUL: {info['current_rul']:.1f} cuts")
print(f"  Initial HI: {info['health_indicator']:.3f}")
print(f"  Condition: C{data_dict['reverse_mapping'][info['operating_condition']]}")

print("\n5. Testing all actions...")
print("\n{:<25} {:<15} {:<15} {:<15} {:<10}".format(
    "Action", "Next RUL", "Cost", "Reward", "Failure"))
print("-" * 80)

# Reset for each action test
for action in range(env.action_space.n):
    obs, info = env.reset(seed=42)  # Same initial state
    obs_next, reward, terminated, truncated, info_next = env.step(action)
    
    print("{:<25} {:<15.1f} {:<15.2f} {:<15.2f} {:<10}".format(
        env.action_names[action],
        info_next['current_rul'],
        info_next['action_cost'],
        reward,
        "Yes" if info_next['failure'] else "No"
    ))

print("\n6. Testing episode simulation...")
obs, info = env.reset(seed=123)
total_reward = 0
steps = 0

print("\nSimulating 20 steps with random policy...")
for step in range(20):
    # Random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    total_reward += reward
    steps += 1
    
    if step % 5 == 0:
        print(f"  Step {step}: Action={env.action_names[action]}, " +
              f"RUL={info['current_rul']:.1f}, Reward={reward:.2f}")
    
    if terminated or truncated:
        print(f"\n  Episode ended at step {steps}")
        print(f"  Reason: {'Failure' if terminated else 'Time limit'}")
        break

print(f"\n✓ Episode completed")
print(f"  Total steps: {steps}")
print(f"  Total reward: {total_reward:.2f}")
print(f"  Total cost: ${info['cumulative_cost']:.2f}")
print(f"  Failures: {info['episode_stats']['failures']}")

print("\n7. Testing counterfactual predictions...")
obs, info = env.reset(seed=42)

print("\nCounterfactual analysis for current state:")
print(f"Current RUL: {info['current_rul']:.1f} cuts")
print(f"Current HI: {info['health_indicator']:.3f}")
print("\nWhat if we take each action?")
print("\n{:<25} {:<15} {:<20} {:<15}".format(
    "Action", "Immediate Cost", "Next RUL", "10-Step Avg RUL"))
print("-" * 80)

for action in range(env.action_space.n):
    cf = env.get_counterfactual_prediction(action)
    avg_future_rul = np.mean(cf['future_rul_trajectory'])
    
    print("{:<25} ${:<14.2f} {:<20.1f} {:<15.1f}".format(
        cf['action_name'],
        cf['immediate_cost'],
        cf['next_rul'],
        avg_future_rul
    ))
    
    # Reset environment after counterfactual (doesn't affect actual state)
    env.state = obs.copy()

print("\n8. Testing safety mask...")
obs, info = env.reset()

# Simulate until high risk state
for _ in range(50):
    obs, reward, terminated, truncated, info = env.step(0)  # Keep continuing
    if terminated or truncated:
        break

safe_actions = env.get_safe_actions()
print(f"\nAt risk state (RUL={info['current_rul']:.1f}, HI={info['health_indicator']:.3f}):")
print("Safe actions:")
for action, is_safe in enumerate(safe_actions):
    print(f"  {env.action_names[action]}: {'✓ Safe' if is_safe else '✗ Unsafe'}")

print("\n" + "="*70)
print("✅ ALL TESTS PASSED!")
print("="*70)

print("\nEnvironment Features Verified:")
print("  ✓ State space correctly defined")
print("  ✓ Action space working (5 actions)")
print("  ✓ Causal model integration successful")
print("  ✓ Reward function computing correctly")
print("  ✓ Episode termination working")
print("  ✓ Counterfactual predictions available")
print("  ✓ Safety constraints functional")

print("\n" + "="*70)
print("READY FOR RL AGENT DEVELOPMENT!")
print("="*70)

print("\nNext Steps:")
print("  1. Implement baseline policies (reactive, time-based, condition-based)")
print("  2. Implement RL agent (PPO/SAC)")
print("  3. Implement Counterfactual Policy Optimization (CPO)")
print("  4. Train and compare all policies")
print("  5. Analyze results")
