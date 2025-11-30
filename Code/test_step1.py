from step1_data_loader import PHM2010DataLoader

# CHANGE THIS to your actual data path
data_path = r"E:\Collaboration Work\With Farooq\phm dataset\PHM Challange 2010 Milling"

# Load data
loader = PHM2010DataLoader(data_path)
data = loader.prepare_data()

print("\n✅ SUCCESS! Data loaded correctly!")
print(f"Train samples: {len(data['train'][0])}")
print(f"Test samples: {len(data['test'][0])}")