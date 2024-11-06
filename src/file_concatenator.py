# %%
import tkinter as tk
from tkinter import filedialog
import sys
import os


# %% Define the main function to handle file concatenation
def concatenate_files(output_path, input_paths):
    with open(output_path, "w") as outfile:
        for file_path in input_paths:
            if os.path.isfile(file_path):
                try:
                    with open(file_path, "r") as infile:
                        content = infile.read()
                        outfile.write("'''\n")
                        outfile.write(content)
                        outfile.write("\n'''\n")
                except UnicodeDecodeError:
                    print(f"Skipping binary file: {file_path}")
            else:
                print(f"File not found: {file_path}")


# %% Define a function to handle multi-step UI file selection
def select_files_via_ui():
    root = tk.Tk()
    root.withdraw()
    selected_files = []

    while True:
        # Let user select files in batches
        file_paths = filedialog.askopenfilenames(
            title="Select Text-Readable Files (Press 'Cancel' when done)",
            filetypes=[("All files", "*.*")],
        )
        if not file_paths:
            break  # Exit loop if user presses "Cancel"
        selected_files.extend(root.tk.splitlist(file_paths))

    return selected_files


# %% Define the main logic to choose between UI and CLI
def main():
    if len(sys.argv) > 1:
        # Command-line mode
        output_file = sys.argv[1]
        input_files = sys.argv[2:]
        concatenate_files(output_file, input_files)
        print(f"Files concatenated into {output_file}")
    else:
        # UI mode
        print("No command-line arguments detected. Using UI for file selection.")
        input_files = select_files_via_ui()
        if input_files:
            output_file = filedialog.asksaveasfilename(
                title="Save Output File",
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            )
            if output_file:
                concatenate_files(output_file, input_files)
                print(f"Files concatenated into {output_file}")
            else:
                print("No output file selected.")
        else:
            print("No input files selected.")


# %% Run the main function
if __name__ == "__main__":
    main()
# %%
