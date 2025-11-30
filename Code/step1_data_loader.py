"""
PHM 2010 Data Loader - Simple Version
Loads wear data and prepares it for training
"""

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler


class PHM2010DataLoader:
    """Load and prepare PHM 2010 milling dataset"""
    
    def __init__(self, base_path):
        """
        Initialize data loader
        
        Args:
            base_path: Path to PHM 2010 data folder
                      e.g., "E:\\Collaboration Work\\With Farooq\\phm dataset\\PHM Challange 2010 Milling"
        """
        self.base_path = base_path
        self.condition_mapping = {1: 0, 4: 1, 6: 2}  # Map conditions to 0,1,2
        self.reverse_mapping = {0: 1, 1: 4, 2: 6}
        
    def load_wear_data(self):
        """Load wear data from all available conditions"""
        
        print("Loading wear data...")
        all_wear_data = []
        available_conditions = []
        
        # Try to load each condition
        for condition in [1, 2, 3, 4, 5, 6]:
            file_path = os.path.join(self.base_path, f'c{condition}', f'c{condition}_wear.csv')
            
            if os.path.exists(file_path):
                print(f"  ✓ Loading condition {condition}")
                wear_df = pd.read_csv(file_path)
                wear_df['condition'] = condition
                all_wear_data.append(wear_df)
                available_conditions.append(condition)
            else:
                print(f"  ✗ Condition {condition} not found (skipping)")
        
        if not all_wear_data:
            raise ValueError("ERROR: No wear data files found! Please check your data path.")
        
        # Combine all data
        combined_wear = pd.concat(all_wear_data, ignore_index=True)
        
        print(f"\n✓ Loaded {len(available_conditions)} conditions: {available_conditions}")
        print(f"  Total samples: {len(combined_wear)}")
        
        return combined_wear, available_conditions
    
    def compute_health_indicator(self, wear_df):
        """
        Compute health indicator (HI) from wear measurements
        HI = average wear across all flutes
        """
        
        # Find flute columns
        flute_cols = [col for col in wear_df.columns if 'flute' in col.lower()]
        
        if not flute_cols:
            raise ValueError("ERROR: No flute columns found in data!")
        
        # Average wear across flutes
        wear_df['health_indicator'] = wear_df[flute_cols].mean(axis=1)
        
        # Normalize HI to [0, 1] for each condition
        for condition in wear_df['condition'].unique():
            mask = wear_df['condition'] == condition
            hi_values = wear_df.loc[mask, 'health_indicator']
            
            hi_min = hi_values.min()
            hi_max = hi_values.max()
            
            if hi_max > hi_min:
                wear_df.loc[mask, 'health_indicator_normalized'] = \
                    (hi_values - hi_min) / (hi_max - hi_min)
            else:
                wear_df.loc[mask, 'health_indicator_normalized'] = 0.5
        
        print(f"✓ Computed health indicators")
        return wear_df
    
    def create_sequences(self, wear_df, sequence_length=20):
        """
        Create temporal sequences for training
        
        Each sequence = 20 time steps of wear measurements
        Label = remaining useful life (RUL) at the end of sequence
        """
        
        print(f"\nCreating sequences (length={sequence_length})...")
        
        sequences = []
        labels = []
        conditions = []
        health_indicators = []
        
        # Process each condition separately
        for condition in wear_df['condition'].unique():
            condition_data = wear_df[wear_df['condition'] == condition].copy()
            condition_data = condition_data.sort_values('cut').reset_index(drop=True)
            
            total_cuts = len(condition_data)
            
            # Create sliding windows
            for start_idx in range(0, total_cuts - sequence_length + 1):
                end_idx = start_idx + sequence_length
                
                # Get sequence window
                sequence_window = condition_data.iloc[start_idx:end_idx]
                
                # Features: flute wear measurements
                flute_cols = [col for col in sequence_window.columns if 'flute' in col.lower()]
                feature_data = sequence_window[flute_cols].values
                
                # Label: remaining cuts after this sequence
                rul = total_cuts - end_idx
                
                # Health indicator at end of sequence
                hi = sequence_window['health_indicator_normalized'].iloc[-1]
                
                # Store
                sequences.append(feature_data)
                labels.append(rul)
                conditions.append(self.condition_mapping.get(condition, 0))
                health_indicators.append(hi)
        
        # Convert to numpy arrays
        sequences = np.array(sequences, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)
        conditions = np.array(conditions, dtype=np.int64)
        health_indicators = np.array(health_indicators, dtype=np.float32)
        
        print(f"✓ Created {len(sequences)} sequences")
        print(f"  Sequence shape: {sequences.shape}")
        print(f"  RUL range: {labels.min():.0f} to {labels.max():.0f} cuts")
        
        return sequences, labels, conditions, health_indicators
    
    def split_data(self, sequences, labels, conditions, health_indicators):
        """Split data into train/validation/test sets"""
        
        n_samples = len(sequences)
        indices = np.arange(n_samples)
        np.random.seed(42)
        np.random.shuffle(indices)
        
        # 70% train, 15% val, 15% test
        train_size = int(n_samples * 0.7)
        val_size = int(n_samples * 0.15)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        
        print(f"\n✓ Split data:")
        print(f"  Train: {len(train_idx)} samples")
        print(f"  Val:   {len(val_idx)} samples")
        print(f"  Test:  {len(test_idx)} samples")
        
        return {
            'train': (sequences[train_idx], labels[train_idx], 
                     conditions[train_idx], health_indicators[train_idx]),
            'val': (sequences[val_idx], labels[val_idx], 
                   conditions[val_idx], health_indicators[val_idx]),
            'test': (sequences[test_idx], labels[test_idx], 
                    conditions[test_idx], health_indicators[test_idx])
        }
    
    def normalize_sequences(self, train_seq, val_seq, test_seq):
        """Normalize sequences using StandardScaler"""
        
        print("\n✓ Normalizing sequences...")
        
        # Reshape for scaling
        n_train, seq_len, n_features = train_seq.shape
        
        train_flat = train_seq.reshape(-1, n_features)
        val_flat = val_seq.reshape(-1, n_features)
        test_flat = test_seq.reshape(-1, n_features)
        
        # Fit scaler on training data only
        scaler = StandardScaler()
        train_flat = scaler.fit_transform(train_flat)
        val_flat = scaler.transform(val_flat)
        test_flat = scaler.transform(test_flat)
        
        # Reshape back
        train_seq = train_flat.reshape(n_train, seq_len, n_features)
        val_seq = val_flat.reshape(len(val_seq), seq_len, n_features)
        test_seq = test_flat.reshape(len(test_seq), seq_len, n_features)
        
        return train_seq, val_seq, test_seq, scaler
    
    def prepare_data(self, sequence_length=20):
        """
        Complete data preparation pipeline
        
        Returns:
            Dictionary with train/val/test data and metadata
        """
        
        print("="*70)
        print("PHM 2010 DATA PREPARATION")
        print("="*70)
        
        # Step 1: Load wear data
        wear_df, available_conditions = self.load_wear_data()
        
        # Step 2: Compute health indicators
        wear_df = self.compute_health_indicator(wear_df)
        
        # Step 3: Create sequences
        sequences, labels, conditions, health_indicators = \
            self.create_sequences(wear_df, sequence_length)
        
        # Step 4: Split data
        data_splits = self.split_data(sequences, labels, conditions, health_indicators)
        
        # Step 5: Normalize
        train_seq, val_seq, test_seq, scaler = self.normalize_sequences(
            data_splits['train'][0],
            data_splits['val'][0],
            data_splits['test'][0]
        )
        
        print("\n" + "="*70)
        print("DATA PREPARATION COMPLETE!")
        print("="*70)
        
        return {
            'train': (train_seq, data_splits['train'][1], 
                     data_splits['train'][2], data_splits['train'][3]),
            'val': (val_seq, data_splits['val'][1],
                   data_splits['val'][2], data_splits['val'][3]),
            'test': (test_seq, data_splits['test'][1],
                    data_splits['test'][2], data_splits['test'][3]),
            'scaler': scaler,
            'condition_mapping': self.condition_mapping,
            'reverse_mapping': self.reverse_mapping,
            'available_conditions': available_conditions
        }


# Test the loader
if __name__ == "__main__":
    print("Testing PHM 2010 Data Loader...")
    print("\nTo use this loader:")
    print("  from phm2010_data_loader import PHM2010DataLoader")
    print("  loader = PHM2010DataLoader('path/to/your/data')")
    print("  data = loader.prepare_data()")
