#%%
import os
import subprocess

# Get all tracked files using git
output = subprocess.check_output(["git", "ls-tree", "-r", "HEAD", "--name-only"], text=True)
file_paths = output.splitlines()

# Build the directory tree
tree = {}
for path in file_paths:
    parts = path.split("/")
    current_level = tree
    for part in parts:
        if part not in current_level:
            current_level[part] = {}
        current_level = current_level[part]

# Function to print the tree with branches and sub-branches
def print_tree(d, prefix=""):
    """Recursively prints the tree structure."""
    branches = list(d.keys())
    for i, branch in enumerate(branches):
        # Use ├── for branches and └── for the last branch
        connector = "└── " if i == len(branches) - 1 else "├── "
        print(f"{prefix}{connector}{branch}")
        if isinstance(d[branch], dict) and d[branch]:
            # Use │   for branches and space for the last branch
            extension = "    " if i == len(branches) - 1 else "│   "
            print_tree(d[branch], prefix + extension)

# Print the directory tree
print_tree(tree)
#%%