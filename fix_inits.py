import os

# The list of folders that need to be treated as Python packages
folders_needing_init = [
    "src",
    "src/preprocessing",
    "src/ocr",
    "src/models",
    "src/train",
    "src/eval"
]

print("Adding missing __init__.py files...")

for folder in folders_needing_init:
    # 1. Ensure the folder exists just in case
    os.makedirs(folder, exist_ok=True)
    
    # 2. Define the path for the __init__.py file
    init_path = os.path.join(folder, "__init__.py")
    
    # 3. Create the empty file
    with open(init_path, "w") as f:
        pass  # Leaves the file completely empty (0 bytes)
        
    print(f"Created: {init_path}")

print("\nFix complete! Your directories are now proper Python packages.")