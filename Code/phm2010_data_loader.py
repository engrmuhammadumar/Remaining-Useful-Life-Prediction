"""
Data Loading and Preprocessing for PHM 2010 Milling Dataset
Handles wear data loading, feature extraction, and sequence generation
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class PHM2010DataLoader:
    """Load and preprocess PHM 2010 milling dataset"""
    
    def __init__(self, base_path):
        self.base_path = base_path
        self.condition_mapping = {1: 0, 4: 1, 6: 2}  # Map to 0-indexed
        self.reverse_mapping = {0: 1, 1: 4, 2: 6}
        
    def load_wear_data(self):
        """Load wear data from all available conditions"""
        
        all_wear_data = []
        available_conditions = []
        
        for condition in [1, 2, 3, 4, 5, 6]:
            condition_path = os.path.join(self.base_path, f'c{condition}', f'c{condition}_wear.csv')
            
            if os.path.exists(condition_path):
                print(f"Loading condition {condition}...")
                wear_df = pd.read_csv(condition_path)
                wear_df['condition'] = condition
                all_wear_data.append(wear_df)
                available_conditions.append(condition)
            else:
                print(f"[WARN] File not found for condition {condition}: {condition_path}")
        
        if not all_wear_data:
            raise ValueError("No wear data files found!")
        
        # Combine all wear data
        combined_wear = pd.concat(all_wear_data, ignore_index=True)
        
        print(f"\nLoaded {len(all_wear_data)} conditions: {available_conditions}")
        print(f"Total wear records: {len(combined_wear)}")
        print(f"\nWear data summary:")
        print(combined_wear.describe())
        print(f"\nRecords per condition:")
        print(combined_wear['condition'].value_counts().sort_index())
        
        return combined_wear, available_conditions
    
    def compute_health_indicator(self, wear_df):
        """
        Compute health indicator (HI) from flute wear measurements.
        HI represents the current degradation state of the tool.
        """
        
        # Average wear across all flutes
        flute_cols = [col for col in wear_df.columns if 'flute' in col.lower()]
        
        if flute_cols:
            wear_df['health_indicator'] = wear_df[flute_cols].mean(axis=1)
        else:
            # Fallback: use any available wear measurement
            print("[WARN] No flute columns found, using first numeric column as HI")
            numeric_cols = wear_df.select_dtypes(include=[np.number]).columns
            wear_df['health_indicator'] = wear_df[numeric_cols[0]]
        
        # Normalize HI to [0, 1] range for each condition separately
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
        
        return wear_df
    
    def create_temporal_sequences(self, wear_df, sequence_length=20, stride=1):
        """
        Create temporal sequences for RUL prediction.
        
        For each tool run:
        - Create sliding windows of length sequence_length
        - Label is the RUL at the end of the sequence
        - Include condition and health indicator information
        """
        
        sequences = []
        labels = []
        conditions = []
        health_indicators = []
        
        # Group by condition (each condition represents independent tool runs)
        for condition in wear_df['condition'].unique():
            condition_data = wear_df[wear_df['condition'] == condition].copy()
            condition_data = condition_data.sort_values('cut').reset_index(drop=True)
            
            total_cuts = len(condition_data)
            
            # Create sequences
            for start_idx in range(0, total_cuts - sequence_length + 1, stride):
                end_idx = start_idx + sequence_length
                
                # Extract sequence window
                sequence_window = condition_data.iloc[start_idx:end_idx]
                
                # Features: wear measurements (flute_1, flute_2, flute_3, etc.)
                flute_cols = [col for col in sequence_window.columns if 'flute' in col.lower()]
                
                if not flute_cols:
                    # Use health_indicator if no flute columns
                    feature_data = sequence_window[['health_indicator']].values
                else:
                    feature_data = sequence_window[flute_cols].values
                
                # RUL label: remaining cuts from the end of the sequence
                rul = total_cuts - end_idx
                
                # Health indicator at the end of sequence
                hi = sequence_window['health_indicator_normalized'].iloc[-1]
                
                sequences.append(feature_data)
                labels.append(rul)
                
                # Map condition to 0-indexed
                if condition in self.condition_mapping:
                    conditions.append(self.condition_mapping[condition])
                else:
                    conditions.append(0)  # Default
                
                health_indicators.append(hi)
        
        sequences = np.array(sequences, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)
        conditions = np.array(conditions, dtype=np.int64)
        health_indicators = np.array(health_indicators, dtype=np.float32)
        
        print(f"\nCreated {len(sequences)} sequences")
        print(f"Sequence shape: {sequences.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Conditions shape: {conditions.shape}")
        print(f"Health indicators shape: {health_indicators.shape}")
        
        # Print statistics
        print(f"\nRUL statistics:")
        print(f"  Min: {labels.min():.1f}, Max: {labels.max():.1f}")
        print(f"  Mean: {labels.mean():.1f}, Std: {labels.std():.1f}")
        
        print(f"\nSequences per condition:")
        for cond_idx in np.unique(conditions):
            count = np.sum(conditions == cond_idx)
            orig_cond = self.reverse_mapping.get(cond_idx, cond_idx)
            print(f"  Condition {orig_cond}: {count} sequences")
        
        return sequences, labels, conditions, health_indicators
    
    def normalize_sequences(self, train_sequences, val_sequences, test_sequences):
        """Normalize sequence data using StandardScaler fitted on training data"""
        
        # Reshape for scaling: (n_samples * seq_len, n_features)
        n_train, seq_len, n_features = train_sequences.shape
        n_val = val_sequences.shape[0]
        n_test = test_sequences.shape[0]
        
        train_flat = train_sequences.reshape(-1, n_features)
        val_flat = val_sequences.reshape(-1, n_features)
        test_flat = test_sequences.reshape(-1, n_features)
        
        # Fit scaler on training data
        scaler = StandardScaler()
        train_flat_scaled = scaler.fit_transform(train_flat)
        val_flat_scaled = scaler.transform(val_flat)
        test_flat_scaled = scaler.transform(test_flat)
        
        # Reshape back
        train_sequences_scaled = train_flat_scaled.reshape(n_train, seq_len, n_features)
        val_sequences_scaled = val_flat_scaled.reshape(n_val, seq_len, n_features)
        test_sequences_scaled = test_flat_scaled.reshape(n_test, seq_len, n_features)
        
        return train_sequences_scaled, val_sequences_scaled, test_sequences_scaled, scaler
    
    def split_data(self, sequences, labels, conditions, health_indicators, 
                   train_ratio=0.7, val_ratio=0.15, random_state=42):
        """Split data into train/val/test sets"""
        
        n_samples = len(sequences)
        indices = np.arange(n_samples)
        np.random.seed(random_state)
        np.random.shuffle(indices)
        
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        
        train_seq = sequences[train_idx]
        val_seq = sequences[val_idx]
        test_seq = sequences[test_idx]
        
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        test_labels = labels[test_idx]
        
        train_cond = conditions[train_idx]
        val_cond = conditions[val_idx]
        test_cond = conditions[test_idx]
        
        train_hi = health_indicators[train_idx]
        val_hi = health_indicators[val_idx]
        test_hi = health_indicators[test_idx]
        
        print(f"\nData split:")
        print(f"  Train: {len(train_seq)} samples")
        print(f"  Val: {len(val_seq)} samples")
        print(f"  Test: {len(test_seq)} samples")
        
        return (train_seq, train_labels, train_cond, train_hi,
                val_seq, val_labels, val_cond, val_hi,
                test_seq, test_labels, test_cond, test_hi)
    
    def prepare_data(self, sequence_length=20, stride=1, train_ratio=0.7, val_ratio=0.15):
        """Complete data preparation pipeline"""
        
        print("="*60)
        print("PHM 2010 Data Preparation Pipeline")
        print("="*60)
        
        # Load wear data
        wear_df, available_conditions = self.load_wear_data()
        
        # Compute health indicator
        wear_df = self.compute_health_indicator(wear_df)
        
        # Create sequences
        sequences, labels, conditions, health_indicators = self.create_temporal_sequences(
            wear_df, sequence_length=sequence_length, stride=stride
        )
        
        # Split data
        (train_seq, train_labels, train_cond, train_hi,
         val_seq, val_labels, val_cond, val_hi,
         test_seq, test_labels, test_cond, test_hi) = self.split_data(
            sequences, labels, conditions, health_indicators,
            train_ratio=train_ratio, val_ratio=val_ratio
        )
        
        # Normalize sequences
        train_seq, val_seq, test_seq, scaler = self.normalize_sequences(
            train_seq, val_seq, test_seq
        )
        
        print("\n" + "="*60)
        print("Data preparation complete!")
        print("="*60)
        
        data_dict = {
            'train': (train_seq, train_labels, train_cond, train_hi),
            'val': (val_seq, val_labels, val_cond, val_hi),
            'test': (test_seq, test_labels, test_cond, test_hi),
            'scaler': scaler,
            'condition_mapping': self.condition_mapping,
            'reverse_mapping': self.reverse_mapping,
            'available_conditions': available_conditions
        }
        
        return data_dict


def load_phm2010_alternative_format(base_path):
    """
    Alternative loader if the dataset is in a different format
    (e.g., separate CSV files per cut or different naming convention)
    """
    
    print(f"Attempting alternative data loading from: {base_path}")
    
    # Try to find any CSV files
    csv_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {base_path}")
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Try to load and inspect the first file
    first_file = csv_files[0]
    print(f"\nInspecting first file: {first_file}")
    
    df = pd.read_csv(first_file)
    print(f"Columns: {df.columns.tolist()}")
    print(f"Shape: {df.shape}")
    print(f"First few rows:")
    print(df.head())
    
    return csv_files, df


if __name__ == "__main__":
    # Example usage
    print("PHM 2010 Data Loader Module")
    print("This module provides data loading and preprocessing functions")
    print("\nTo use:")
    print("  from phm2010_data_loader import PHM2010DataLoader")
    print("  loader = PHM2010DataLoader('path/to/phm/data')")
    print("  data_dict = loader.prepare_data()")
