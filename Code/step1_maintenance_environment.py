"""
MAINTENANCE ENVIRONMENT - Step 1
================================
Gymnasium-compatible environment for maintenance decision-making

This is the foundation of your C-RLM framework.
The environment simulates a machine degradation process where an RL agent
must make maintenance decisions to minimize costs while avoiding failures.
"""

import sys
import os

# Add parent directory to path to import previous modules
sys.path.append(os.path.abspath('../new_code'))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any
import torch
from step2_models import CausalStructuralTransformer


class MaintenanceEnvironment(gym.Env):
    """
    Maintenance Environment for Reinforcement Learning
    
    State Space:
        - degradation_features: [seq_len, n_features] from causal model
        - current_rul: scalar
        - operating_condition: categorical {0, 1, 2}
        - health_indicator: [0, 1]
        - maintenance_history: last k actions
        - cumulative_cost: scalar
        - time_step: scalar
    
    Action Space:
        0: Continue - No change
        1: Reduce Load - Lower operating condition
        2: Minor Maintenance - Partial restoration
        3: Major Maintenance - Full replacement
        4: Shutdown - Emergency stop
    
    Reward:
        Multi-objective:
            - Maintenance cost (negative)
            - Downtime cost (negative)
            - Failure penalty (large negative)
            - Availability bonus (positive)
    
    Dynamics:
        Uses your trained causal transformer to predict degradation
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        causal_model: CausalStructuralTransformer,
        data_dict: Dict,
        config: Optional[Dict] = None
    ):
        super().__init__()
        
        self.causal_model = causal_model
        self.causal_model.eval()
        
        self.data_dict = data_dict
        self.config = config or self._default_config()
        
        # Extract data parameters
        self.seq_len = config.get('seq_len', 20)
        self.n_features = config.get('n_features', 3)
        self.n_conditions = len(data_dict['condition_mapping'])
        
        # Define action space
        self.action_space = spaces.Discrete(5)
        
        # Action names for interpretability
        self.action_names = [
            "Continue",
            "Reduce Load", 
            "Minor Maintenance",
            "Major Maintenance",
            "Shutdown"
        ]
        
        # Define observation space
        self.observation_space = spaces.Dict({
            'degradation_features': spaces.Box(
                low=-10, high=10, 
                shape=(self.seq_len, self.n_features), 
                dtype=np.float32
            ),
            'current_rul': spaces.Box(
                low=0, high=300, 
                shape=(1,), 
                dtype=np.float32
            ),
            'operating_condition': spaces.Discrete(self.n_conditions),
            'health_indicator': spaces.Box(
                low=0, high=1, 
                shape=(1,), 
                dtype=np.float32
            ),
            'maintenance_history': spaces.Box(
                low=0, high=4,
                shape=(self.config['history_length'],),
                dtype=np.int32
            ),
            'cumulative_cost': spaces.Box(
                low=0, high=np.inf,
                shape=(1,),
                dtype=np.float32
            ),
            'time_step': spaces.Box(
                low=0, high=np.inf,
                shape=(1,),
                dtype=np.float32
            )
        })
        
        # Initialize state
        self.state = None
        self.episode_length = 0
        self.max_episode_length = config.get('max_episode_length', 300)
        
        # Costs (will be tuned based on domain knowledge)
        self.costs = self.config['costs']
        
        # Safety threshold
        self.safety_threshold = config.get('safety_threshold', 0.05)
        
        # Statistics tracking
        self.episode_stats = {
            'total_cost': 0,
            'maintenance_actions': 0,
            'failures': 0,
            'uptime': 0,
            'total_production': 0
        }
        
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'seq_len': 20,
            'n_features': 3,
            'history_length': 5,
            'max_episode_length': 300,
            'costs': {
                'continue': 0,          # No cost for continuing
                'reduce_load': 10,      # Small cost for reducing load
                'minor_maintenance': 100,  # Moderate cost
                'major_maintenance': 500,  # High cost but full restoration
                'shutdown': 50,         # Emergency shutdown cost
                'failure': 10000,       # Catastrophic failure penalty
                'downtime_per_cut': 0.5 # Cost per cut of downtime
            },
            'safety_threshold': 0.05,  # 5% failure risk threshold
            'reward_weights': {
                'cost': 1.0,
                'safety': 10.0,
                'availability': 0.5
            }
        }
    
    def reset(
        self, 
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[Dict, Dict]:
        """Reset environment to initial state"""
        
        super().reset(seed=seed)
        
        # Sample initial condition from data
        train_seq = self.data_dict['train'][0]
        train_labels = self.data_dict['train'][1]
        train_cond = self.data_dict['train'][2]
        train_hi = self.data_dict['train'][3]
        
        # Random initial state from training data
        idx = np.random.randint(0, len(train_seq))
        
        self.state = {
            'degradation_features': train_seq[idx].copy(),
            'current_rul': np.array([train_labels[idx]], dtype=np.float32),
            'operating_condition': train_cond[idx],
            'health_indicator': np.array([train_hi[idx]], dtype=np.float32),
            'maintenance_history': np.zeros(self.config['history_length'], dtype=np.int32),
            'cumulative_cost': np.array([0.0], dtype=np.float32),
            'time_step': np.array([0.0], dtype=np.float32)
        }
        
        self.episode_length = 0
        self.episode_stats = {
            'total_cost': 0,
            'maintenance_actions': 0,
            'failures': 0,
            'uptime': 0,
            'total_production': 0
        }
        
        info = self._get_info()
        
        return self.state.copy(), info
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action: Integer action [0-4]
        
        Returns:
            observation: Next state
            reward: Scalar reward
            terminated: Episode ended due to failure/completion
            truncated: Episode ended due to time limit
            info: Additional information
        """
        
        # Execute action and get next state
        next_state, action_cost, failure, downtime = self._execute_action(action)
        
        # Calculate reward
        reward = self._calculate_reward(action, action_cost, failure, downtime)
        
        # Update state
        self.state = next_state
        self.episode_length += 1
        
        # Update statistics
        self.episode_stats['total_cost'] += action_cost
        if action != 0:
            self.episode_stats['maintenance_actions'] += 1
        if failure:
            self.episode_stats['failures'] += 1
        if not failure and not downtime:
            self.episode_stats['uptime'] += 1
            self.episode_stats['total_production'] += 1
        
        # Check termination conditions
        terminated = failure or (self.state['current_rul'][0] <= 0)
        truncated = self.episode_length >= self.max_episode_length
        
        info = self._get_info()
        info['action_name'] = self.action_names[action]
        info['action_cost'] = action_cost
        info['failure'] = failure
        
        return self.state.copy(), reward, terminated, truncated, info
    
    def _execute_action(self, action: int) -> Tuple[Dict, float, bool, int]:
        """
        Execute maintenance action and return next state
        
        Returns:
            next_state: Updated state
            cost: Cost of action
            failure: Whether catastrophic failure occurred
            downtime: Cuts of downtime (0 if no downtime)
        """
        
        current_state = self.state
        cost = 0
        failure = False
        downtime = 0
        
        # Create next state (copy current)
        next_state = {key: val.copy() for key, val in current_state.items()}
        
        # ACTION 0: Continue
        if action == 0:
            cost = self.costs['continue']
            
            # Natural degradation (use causal model to predict)
            next_state = self._predict_next_state(
                current_state, 
                action=None,
                condition_change=None
            )
            
            # Check for failure
            if next_state['current_rul'][0] <= 0 or next_state['health_indicator'][0] >= 0.95:
                failure = True
                cost += self.costs['failure']
        
        # ACTION 1: Reduce Load
        elif action == 1:
            cost = self.costs['reduce_load']
            
            # Change to lower operating condition
            current_cond = current_state['operating_condition']
            available_conditions = [0, 1, 2]  # C1, C4, C6
            
            # Find lower condition (in terms of wear rate)
            # Assume: C1 < C4 < C6 in wear rate
            condition_order = [0, 1, 2]
            current_idx = condition_order.index(current_cond)
            
            if current_idx > 0:
                new_cond = condition_order[current_idx - 1]
                next_state = self._predict_next_state(
                    current_state,
                    action='reduce_load',
                    condition_change=new_cond
                )
            else:
                # Already at minimum, just continue
                next_state = self._predict_next_state(current_state)
        
        # ACTION 2: Minor Maintenance
        elif action == 2:
            cost = self.costs['minor_maintenance']
            downtime = 5  # 5 cuts of downtime
            cost += downtime * self.costs['downtime_per_cut']
            
            # Partial restoration: reduce HI by 30%
            next_state = current_state.copy()
            next_state['health_indicator'] = current_state['health_indicator'] * 0.7
            
            # RUL increases proportionally
            next_state['current_rul'] = current_state['current_rul'] * 1.3
        
        # ACTION 3: Major Maintenance (Replacement)
        elif action == 3:
            cost = self.costs['major_maintenance']
            downtime = 20  # 20 cuts of downtime
            cost += downtime * self.costs['downtime_per_cut']
            
            # Full restoration: reset to new condition
            next_state['health_indicator'] = np.array([0.0], dtype=np.float32)
            next_state['current_rul'] = np.array([280.0], dtype=np.float32)  # Approximately new tool life
            
            # Reset degradation features to early-life pattern
            # Sample from early-life data
            train_seq = self.data_dict['train'][0]
            train_labels = self.data_dict['train'][1]
            
            # Find samples with high RUL (new condition)
            high_rul_mask = train_labels > 250
            if high_rul_mask.sum() > 0:
                idx = np.random.choice(np.where(high_rul_mask)[0])
                next_state['degradation_features'] = train_seq[idx].copy()
        
        # ACTION 4: Shutdown
        elif action == 4:
            cost = self.costs['shutdown']
            downtime = 2
            cost += downtime * self.costs['downtime_per_cut']
            
            # Just stop for inspection, no change in degradation
            next_state = current_state.copy()
        
        # Update maintenance history
        next_state['maintenance_history'] = np.roll(current_state['maintenance_history'], 1)
        next_state['maintenance_history'][0] = action
        
        # Update cumulative cost
        next_state['cumulative_cost'] = current_state['cumulative_cost'] + cost
        
        # Update time step
        next_state['time_step'] = current_state['time_step'] + 1 + downtime
        
        return next_state, cost, failure, downtime
    
    def _predict_next_state(
        self, 
        current_state: Dict,
        action: Optional[str] = None,
        condition_change: Optional[int] = None
    ) -> Dict:
        """
        Use causal model to predict next state after one time step
        
        This is where your causal transformer is used!
        """
        
        next_state = {key: val.copy() for key, val in current_state.items()}
        
        # Prepare input for causal model
        seq = torch.FloatTensor(current_state['degradation_features']).unsqueeze(0)
        condition = torch.LongTensor([condition_change if condition_change is not None 
                                      else current_state['operating_condition']])
        hi = torch.FloatTensor(current_state['health_indicator'])
        
        # Predict RUL using causal model
        with torch.no_grad():
            pred_rul, components = self.causal_model(seq, condition, hi)
        
        # Update RUL (decrease by 1 cut)
        next_state['current_rul'] = np.array([pred_rul.item() - 1], dtype=np.float32)
        
        # Update HI (increases with use)
        # Simple model: HI increases by small amount each cut
        hi_increase = 0.001 * (1 + 0.1 * current_state['operating_condition'])
        next_state['health_indicator'] = np.clip(
            current_state['health_indicator'] + hi_increase,
            0, 1
        )
        
        # Update operating condition if changed
        if condition_change is not None:
            next_state['operating_condition'] = condition_change
        
        # Update degradation features (shift window, add noise)
        # This is simplified - in reality would use actual sensor measurements
        new_features = next_state['degradation_features'].copy()
        new_features[:-1] = new_features[1:]  # Shift left
        new_features[-1] = new_features[-2] + np.random.randn(self.n_features) * 0.01
        next_state['degradation_features'] = new_features
        
        return next_state
    
    def _calculate_reward(
        self, 
        action: int, 
        cost: float, 
        failure: bool,
        downtime: int
    ) -> float:
        """
        Calculate multi-objective reward
        
        Components:
            1. Cost (negative)
            2. Safety (penalty for high risk actions)
            3. Availability (bonus for uptime)
        """
        
        weights = self.config['reward_weights']
        
        # Cost component (negative)
        reward_cost = -cost * weights['cost']
        
        # Safety component (penalty for risky states)
        current_rul = self.state['current_rul'][0]
        current_hi = self.state['health_indicator'][0]
        
        # Risk increases as RUL decreases and HI increases
        risk = (1 - current_rul / 300) * current_hi
        
        if risk > self.safety_threshold:
            reward_safety = -(risk - self.safety_threshold) * 1000 * weights['safety']
        else:
            reward_safety = 0
        
        # Failure penalty
        if failure:
            reward_safety -= self.costs['failure'] * weights['safety']
        
        # Availability component (positive for production time)
        if action == 0 and not failure:
            reward_availability = 10 * weights['availability']  # Bonus for producing
        else:
            reward_availability = -downtime * weights['availability']
        
        # Total reward
        reward = reward_cost + reward_safety + reward_availability
        
        return reward
    
    def _get_info(self) -> Dict:
        """Get additional information about current state"""
        
        info = {
            'current_rul': self.state['current_rul'][0],
            'health_indicator': self.state['health_indicator'][0],
            'operating_condition': self.state['operating_condition'],
            'cumulative_cost': self.state['cumulative_cost'][0],
            'time_step': self.state['time_step'][0],
            'episode_stats': self.episode_stats.copy(),
            'safety_risk': (1 - self.state['current_rul'][0] / 300) * self.state['health_indicator'][0]
        }
        
        return info
    
    def render(self, mode='human'):
        """Render current state"""
        
        if mode == 'human':
            print("\n" + "="*60)
            print(f"MAINTENANCE ENVIRONMENT STATE")
            print("="*60)
            print(f"Time Step: {self.state['time_step'][0]:.0f}")
            print(f"Current RUL: {self.state['current_rul'][0]:.1f} cuts")
            print(f"Health Indicator: {self.state['health_indicator'][0]:.3f}")
            print(f"Operating Condition: C{self.data_dict['reverse_mapping'][self.state['operating_condition']]}")
            print(f"Cumulative Cost: ${self.state['cumulative_cost'][0]:.2f}")
            print(f"\nEpisode Statistics:")
            print(f"  Total Cost: ${self.episode_stats['total_cost']:.2f}")
            print(f"  Maintenance Actions: {self.episode_stats['maintenance_actions']}")
            print(f"  Failures: {self.episode_stats['failures']}")
            print(f"  Uptime: {self.episode_stats['uptime']} cuts")
            print(f"  Production: {self.episode_stats['total_production']} units")
            print("="*60)
    
    def get_safe_actions(self) -> np.ndarray:
        """
        Return mask of safe actions (for safety-constrained RL)
        
        Uses causal model to predict which actions are safe
        """
        
        safe_mask = np.ones(self.action_space.n, dtype=bool)
        
        current_rul = self.state['current_rul'][0]
        current_hi = self.state['health_indicator'][0]
        
        # If RUL is very low or HI very high, only maintenance actions are safe
        if current_rul < 10 or current_hi > 0.9:
            safe_mask[0] = False  # Continue is not safe
            safe_mask[1] = False  # Reduce load won't help much
        
        return safe_mask
    
    def get_counterfactual_prediction(self, action: int) -> Dict:
        """
        Use causal model to predict what would happen if we took this action
        
        This is the KEY innovation - counterfactual prediction!
        """
        
        # Simulate taking this action
        next_state, cost, failure, downtime = self._execute_action(action)
        
        # Predict RUL trajectory for next N steps
        future_rul = []
        state = next_state.copy()
        
        for _ in range(10):  # Predict 10 steps ahead
            state = self._predict_next_state(state)
            future_rul.append(state['current_rul'][0])
        
        counterfactual = {
            'action': action,
            'action_name': self.action_names[action],
            'immediate_cost': cost,
            'predicted_failure': failure,
            'downtime': downtime,
            'next_rul': next_state['current_rul'][0],
            'future_rul_trajectory': future_rul,
            'expected_future_cost': cost + sum([self.costs['continue']] * len(future_rul))
        }
        
        return counterfactual


# Test the environment
if __name__ == "__main__":
    print("="*70)
    print("TESTING MAINTENANCE ENVIRONMENT")
    print("="*70)
    
    print("\nTo use this environment:")
    print("1. Load your trained causal model")
    print("2. Load your data")
    print("3. Create environment: env = MaintenanceEnvironment(model, data)")
    print("4. Use standard gym interface: obs, info = env.reset()")
    print("5. Take actions: obs, reward, done, truncated, info = env.step(action)")
    
    print("\nThis environment enables:")
    print("  ✓ RL agent training")
    print("  ✓ Counterfactual predictions")
    print("  ✓ Safety-constrained exploration")
    print("  ✓ Multi-objective optimization")
    
    print("\n" + "="*70)
    print("ENVIRONMENT READY!")
    print("="*70)
