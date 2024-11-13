"""
git_tracked_files_tree.py

This script generates a hierarchical, tree-like structure of all files and directories 
currently tracked by Git, excluding those specified in the .gitignore file. It provides 
an organized view of the project's file system, which is helpful for reviewing the 
directory layout or sharing a clean structure of your project with others.

Usage:
- Ensure you are in the root directory of your Git project.
- Run the script using Python: `python git_tracked_files_tree.py`
- The output will display a tree-like structure of tracked files and directories.

Dependencies:
- Python 3.x
- subprocess (standard Python library)

Note:
- The script only shows files tracked by Git, respecting your .gitignore settings.
- Useful for preparing a clean project view for documentation or sharing purposes.
"""
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