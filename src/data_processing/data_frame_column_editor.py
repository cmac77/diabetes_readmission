"""
data_frame_column_editor.py

This module provides utilities for editing the columns of pandas DataFrames. It includes functions to rename columns,
add new columns, and modify column values based on user specifications. This is helpful for preparing data for analysis
or synthetic data generation.

Functions:
    - add_column(df: pd.DataFrame, column_name: str, values: List): Adds a new column to the DataFrame.
    - rename_columns(df: pd.DataFrame, name_map: Dict[str, str]): Renames columns in the DataFrame according to a mapping.

Example Usage:
    # Add a new column:
    df = add_column(df, 'new_column', [1, 2, 3, 4])
    
    # Rename existing columns:
    renamed_df = rename_columns(df, {'old_name1': 'new_name1', 'old_name2': 'new_name2'})
    
    # Example Output:
    print(renamed_df.head())

Requirements:
    pandas, tkinter, numpy
"""

import tkinter as tk
import pandas as pd
import numpy as np


class DataFrameColumnEditor:
    def __init__(
        self,
        df,
        window_width_ratio=0.25,
        window_height_ratio=0.25,
        top_panel_height_ratio=0.1,
        bottom_panel_height_ratio=0.25,
        root_bg_color="lightgray",
        top_panel_color="lightgray",
        middle_panel_color="lightgray",
        bottom_panel_color="lightgray",
        listbox_top_color="white",
        listbox_bottom_color="white",
        button_frame_color="lightgray",
        button_bg_color="lightgray",
        button_fg_color="black",
        divider_color="lightgray",
        scrollbar_color="lightgray",
    ):
        self.df = df
        self.current_column_index = 0
        self.root = tk.Tk()
        self.root.title("Column Editor")
        self.is_editing = tk.BooleanVar(value=False)

        # Make the window stay on top of other windows
        self.root.attributes("-topmost", True)

        # Handle window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Store pending changes for column names and unique values
        self.top_listbox_edits = {}  # To store column name edits
        self.bottom_listbox_edits = (
            {}
        )  # To store unique value edits for each column

        # Set default or user-specified panel sizes and colors
        self.window_width_ratio = window_width_ratio
        self.window_height_ratio = window_height_ratio
        self.top_panel_height_ratio = top_panel_height_ratio
        self.bottom_panel_height_ratio = bottom_panel_height_ratio
        self.root_bg_color = (
            root_bg_color  # Add root window background color initialization
        )
        self.top_panel_color = top_panel_color
        self.middle_panel_color = middle_panel_color
        self.bottom_panel_color = bottom_panel_color
        self.listbox_top_color = listbox_top_color
        self.listbox_bottom_color = listbox_bottom_color
        self.button_frame_color = button_frame_color
        self.button_bg_color = button_bg_color
        self.button_fg_color = button_fg_color
        self.divider_color = divider_color
        self.scrollbar_color = scrollbar_color

        self.create_ui_window()

    def create_ui_window(self):
        """Create the overall UI window by calling helper functions."""
        self.configure_root_window()

        # Set window dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = int(screen_width * self.window_width_ratio)
        total_height = int(screen_height * self.window_height_ratio)

        self.root.geometry(f"{window_width}x{total_height}+50+50")

        # Divide the total height between the top, middle, and bottom panels
        self.top_panel_height = int(total_height * self.top_panel_height_ratio)
        self.bottom_panel_height = int(
            total_height * self.bottom_panel_height_ratio
        )
        self.middle_panel_height = total_height - (
            self.top_panel_height + self.bottom_panel_height
        )

        # Create and configure panels
        self.create_top_panel()
        self.create_middle_panel()
        self.create_bottom_panel()

        # Load the first column
        self.load_column(0)

        # Start the Tkinter event loop
        self.root.mainloop()

    def configure_root_window(self):
        """Configure the root window settings like background color."""
        self.root.configure(bg=self.root_bg_color)

    def create_top_panel(self):
        """Create the top panel with buttons."""
        top_panel = tk.Frame(
            self.root, height=self.top_panel_height, bg=self.top_panel_color
        )
        top_panel.pack(fill="x", expand=False)

        button_frame = tk.Frame(
            top_panel, bg=self.button_frame_color
        )  # Frame color for buttons
        button_frame.place(
            relx=0.5, rely=0.5, anchor="center"
        )  # Explicitly centering the frame

        self.next_button = tk.Button(
            button_frame,
            text="Next Column",
            command=self.next_column,
            bg=self.button_bg_color,
            fg=self.button_fg_color,  # Button background and foreground colors
        )
        self.next_button.pack(side="left", padx=10)

        self.confirm_button = tk.Button(
            button_frame,
            text="Confirm Text Edit",
            command=self.confirm_changes,
            bg=self.button_bg_color,
            fg=self.button_fg_color,  # Button background and foreground colors
        )
        self.confirm_button.pack(side="left", padx=10)

        self.recode_button = tk.Button(
            button_frame,
            text="Recode as Ordinal",
            command=self.recode_values_as_ordinal,
            bg=self.button_bg_color,
            fg=self.button_fg_color,  # Button background and foreground colors
        )
        self.recode_button.pack(side="left", padx=10)

    def create_middle_panel(self):
        """Create the middle panel with two listboxes and a divider."""
        middle_panel = tk.Frame(
            self.root,
            height=self.middle_panel_height,
            bg=self.middle_panel_color,
        )
        middle_panel.pack(fill="x", expand=True)

        # Divide the middle panel into two listboxes: top for column name, bottom for unique values
        top_listbox_height = int(
            self.middle_panel_height * 0.2
        )  # 20% for column name
        bottom_listbox_height = (
            self.middle_panel_height - top_listbox_height
        )  # 80% for unique values

        # Top Listbox (for column name)
        self.top_listbox_frame = tk.Frame(
            middle_panel, height=top_listbox_height, bg=self.middle_panel_color
        )
        self.top_listbox_frame.pack(fill="x")
        self.top_listbox = tk.Listbox(
            self.top_listbox_frame,
            selectmode="extended",
            height=1,
            bg=self.listbox_top_color,
            bd=1,
            relief="solid",  # Background color, border width, border style
            highlightthickness=0,
            highlightbackground=None,  # Color
            highlightcolor=None,  # Border thickness and colors
        )
        self.top_listbox.pack(fill="both", expand=True)
        self.top_listbox.bind("<Double-1>", self.edit_top_listbox)

        # Divider between the two listboxes
        divider = tk.Frame(middle_panel, height=3, bg=self.divider_color)
        divider.pack(fill="x")

        # Bottom Listbox (for unique values)
        self.bottom_listbox_frame = tk.Frame(
            middle_panel,
            height=bottom_listbox_height,
            bg=self.middle_panel_color,
        )
        self.bottom_listbox_frame.pack(fill="x", expand=True)
        self.bottom_listbox = tk.Listbox(
            self.bottom_listbox_frame,
            selectmode="extended",
            height=10,
            bg=self.listbox_bottom_color,
            bd=1,
            relief="solid",  # Background color, border width, border style
            highlightthickness=0,
            highlightbackground=None,  # Color
            highlightcolor=None,  # Border thickness and colors
        )
        self.bottom_listbox.pack(side="left", fill="both", expand=True)
        self.bottom_listbox.bind("<Double-1>", self.edit_bottom_listbox)

        scrollbar = tk.Scrollbar(
            self.bottom_listbox_frame,
            command=self.bottom_listbox.yview,
            bg=self.scrollbar_color,
        )
        scrollbar.pack(side="right", fill="y")
        self.bottom_listbox.config(yscrollcommand=scrollbar.set)

    def create_bottom_panel(self):
        """Create the bottom panel with Move Up and Move Down buttons."""
        bottom_panel = tk.Frame(
            self.root,
            height=self.bottom_panel_height,
            bg=self.bottom_panel_color,
        )
        bottom_panel.pack(fill="x", expand=False)
        bottom_panel.pack_propagate(
            False
        )  # Prevent auto-sizing based on children

        button_frame = tk.Frame(
            bottom_panel, bg=self.button_frame_color
        )  # Frame color for bottom buttons
        button_frame.place(relx=0.5, rely=0.5, anchor="center")

        self.move_up_button = tk.Button(
            button_frame,
            text="Move Up",
            command=self.move_up,
            bg=self.button_bg_color,
            fg=self.button_fg_color,  # Button background and foreground colors
        )
        self.move_up_button.pack(side="left", padx=10)

        self.move_down_button = tk.Button(
            button_frame,
            text="Move Down",
            command=self.move_down,
            bg=self.button_bg_color,
            fg=self.button_fg_color,  # Button background and foreground colors
        )
        self.move_down_button.pack(side="left", padx=10)

    def load_column(self, col_index):
        """Load the specified column and update the listboxes."""
        self.current_column_index = col_index
        col_name = self.df.columns[col_index]

        # Enable the bottom listbox before clearing it (to make sure it can be cleared)
        self.bottom_listbox.config(state=tk.NORMAL)

        # Clear the listboxes before inserting new data
        self.top_listbox.delete(0, tk.END)
        self.bottom_listbox.delete(0, tk.END)

        # Insert column name in the top listbox
        self.top_listbox.insert(tk.END, col_name)

        # Check if the column is numerical
        is_numeric = pd.api.types.is_numeric_dtype(self.df[col_name])

        # If the column is non-numerical, insert unique values
        if not is_numeric:
            unique_values = self.df[col_name].unique()
            for value in unique_values:
                self.bottom_listbox.insert(tk.END, value)
        else:
            self.show_numerical_column_message()  # Handle numerical columns

        self.root.update()  # Force UI update after changes

    def next_column(self):
        next_col_index = (self.current_column_index + 1) % len(self.df.columns)
        self.load_column(next_col_index)

    def move_up(self):
        if self.is_editing.get():
            return

        selected_indices = list(self.bottom_listbox.curselection())
        if not selected_indices or selected_indices[0] == 0:
            return

        for index in selected_indices:
            if index > 0:
                value = self.bottom_listbox.get(index)
                self.bottom_listbox.delete(index)
                self.bottom_listbox.insert(index - 1, value)

        new_selection = [i - 1 for i in selected_indices]
        for index in new_selection:
            self.bottom_listbox.selection_set(index)

    def move_down(self):
        if self.is_editing.get():
            return

        selected_indices = list(self.bottom_listbox.curselection())
        if (
            not selected_indices
            or selected_indices[-1] == self.bottom_listbox.size() - 1
        ):
            return

        for index in reversed(selected_indices):
            if index < self.bottom_listbox.size() - 1:
                value = self.bottom_listbox.get(index)
                self.bottom_listbox.delete(index)
                self.bottom_listbox.insert(index + 1, value)

        new_selection = [i + 1 for i in selected_indices]
        for index in new_selection:
            self.bottom_listbox.selection_set(index)

    def edit_top_listbox(self, event):
        """Allow editing of the selected item in the top listbox."""
        index = self.top_listbox.curselection()[
            0
        ]  # Get the selected item's index
        current_value = self.top_listbox.get(
            index
        )  # Get the current value of the selected item

        # Get the x,y-coordinate of the selected item
        bbox = self.bottom_listbox.bbox(index)
        y_position = bbox[1]
        x_position = bbox[
            0
        ]  # bbox returns the bounding box of the selected item

        # Disable move buttons when entering edit mode
        self.move_up_button.config(state=tk.DISABLED)
        self.move_down_button.config(state=tk.DISABLED)

        # Create an entry widget to edit the value
        self.edit_entry = tk.Entry(self.top_listbox, bd=0)
        self.edit_entry.insert(0, current_value)
        self.edit_entry.place(
            x=x_position, y=y_position
        )  # Adjust the placement using bbox's y-coordinate
        self.edit_entry.focus()

        # Handle saving changes (Enter key)
        self.edit_entry.bind(
            "<Return>", lambda e: self.save_top_listbox_edit(index)
        )

        # Handle exiting without saving changes (Esc key)
        self.edit_entry.bind(
            "<Escape>", lambda e: self.cancel_top_listbox_edit()
        )

        # Handle saving changes when clicking outside the editing box (FocusOut)
        self.edit_entry.bind(
            "<FocusOut>", lambda e: self.save_top_listbox_edit(index)
        )

    def cancel_top_listbox_edit(self):
        """Exit edit mode without saving changes for the top listbox."""
        self.edit_entry.destroy()

        # Re-enable move buttons after editing is finished
        self.move_up_button.config(state=tk.NORMAL)
        self.move_down_button.config(state=tk.NORMAL)

    def save_top_listbox_edit(self, index):
        """Track changes to the top listbox (column names) but do not propagate to the DataFrame yet."""
        new_value = self.edit_entry.get()
        old_value = self.top_listbox.get(index)

        # Update the listbox with the new value
        self.top_listbox.delete(index)
        self.top_listbox.insert(index, new_value)
        self.edit_entry.destroy()

        # Track the column name change, but don't apply it to the DataFrame yet
        self.top_listbox_edits[old_value] = new_value

        # Re-enable move buttons after editing is finished
        self.move_up_button.config(state=tk.NORMAL)
        self.move_down_button.config(state=tk.NORMAL)

    def edit_bottom_listbox(self, event):
        """Allow editing of the selected item in the bottom listbox."""
        index = self.bottom_listbox.curselection()[
            0
        ]  # Get the selected item's index
        current_value = self.bottom_listbox.get(
            index
        )  # Get the current value of the selected item

        # Get the x,y-coordinate of the selected item
        bbox = self.bottom_listbox.bbox(index)
        y_position = bbox[1]
        x_position = bbox[
            0
        ]  # bbox returns the bounding box of the selected item

        # Disable move buttons when entering edit mode
        self.move_up_button.config(state=tk.DISABLED)
        self.move_down_button.config(state=tk.DISABLED)

        # Create an entry widget to edit the value
        self.edit_entry = tk.Entry(self.bottom_listbox, bd=0)
        self.edit_entry.insert(0, current_value)
        self.edit_entry.place(
            x=x_position, y=y_position
        )  # Adjust the placement using bbox's y-coordinate
        self.edit_entry.focus()

        # Handle saving changes (Enter key)
        self.edit_entry.bind(
            "<Return>", lambda e: self.save_bottom_listbox_edit(index)
        )

        # Handle exiting without saving changes (Esc key)
        self.edit_entry.bind(
            "<Escape>", lambda e: self.cancel_bottom_listbox_edit()
        )

        # Handle saving changes when clicking outside the editing box (FocusOut)
        self.edit_entry.bind(
            "<FocusOut>", lambda e: self.save_bottom_listbox_edit(index)
        )

    def cancel_bottom_listbox_edit(self):
        """Exit edit mode without saving changes for the bottom listbox."""
        self.edit_entry.destroy()

        # Re-enable move buttons after editing is finished
        self.move_up_button.config(state=tk.NORMAL)
        self.move_down_button.config(state=tk.NORMAL)

    def save_bottom_listbox_edit(self, index):
        """Track changes to the bottom listbox (unique values), convert empty to NaN, and display 'NaN' in the listbox."""
        new_value = (
            self.edit_entry.get().strip()
        )  # Strip any leading/trailing spaces
        old_value = self.bottom_listbox.get(index)

        # If the new value is empty, treat it as NaN
        if new_value == "":
            new_value = np.nan

        # Update the listbox with the new value (display 'NaN' explicitly for NaN values)
        self.bottom_listbox.delete(index)
        display_value = (
            "NaN" if pd.isna(new_value) else new_value
        )  # Display 'NaN' in the listbox
        self.bottom_listbox.insert(index, display_value)
        self.edit_entry.destroy()

        # Track the unique value change, but don't apply it to the DataFrame yet
        col_name = self.df.columns[self.current_column_index]
        if col_name not in self.bottom_listbox_edits:
            self.bottom_listbox_edits[col_name] = {}

        self.bottom_listbox_edits[col_name][old_value] = new_value

        # Re-enable move buttons after editing is finished
        self.move_up_button.config(state=tk.NORMAL)
        self.move_down_button.config(state=tk.NORMAL)

    def edit_listbox_item(self, listbox, index, is_top=False):
        if self.is_editing.get():
            return

        # Set editing mode
        self.is_editing.set(True)

        # Get bounding box of the selected item
        bbox = listbox.bbox(index)
        if bbox:
            x, y, width, height = bbox

            value = listbox.get(index)
            entry = tk.Entry(listbox, width=width)
            entry.insert(0, value)
            entry.place(x=x, y=y)

            def save_edit(event=None):
                new_value = entry.get().strip()

                if is_top:
                    old_col_name = self.df.columns[self.current_column_index]
                    self.df.rename(
                        columns={old_col_name: new_value}, inplace=True
                    )
                else:
                    if new_value == "":
                        new_value = np.nan  # Convert empty input to NaN
                    old_value = listbox.get(index)

                    # **Direct assignment in DataFrame** to propagate changes correctly
                    col_name = self.df.columns[self.current_column_index]
                    mask = self.df[col_name] == old_value
                    self.df.loc[mask, col_name] = new_value

                listbox.delete(index)
                listbox.insert(index, new_value)
                entry.destroy()
                self.is_editing.set(False)

            def cancel_edit(event=None):
                entry.destroy()
                self.is_editing.set(False)

            entry.bind("<Return>", save_edit)
            entry.bind("<Escape>", cancel_edit)
            entry.bind("<FocusOut>", save_edit)
            entry.focus_set()

    def recode_values_as_ordinal(self):
        """Recode the values in the bottom listbox as ordinal, but first confirm any pending edits."""

        # Force confirmation of any pending text edits before proceeding
        self.confirm_changes()

        col_name = self.df.columns[self.current_column_index]
        is_numeric = pd.api.types.is_numeric_dtype(self.df[col_name])

        if not is_numeric:
            # Get the current order of unique values in the bottom listbox, skipping NaNs
            seen_values = (
                set()
            )  # Track values that have been assigned ordinals
            unique_values = []

            # Traverse the listbox in its current order
            for i in range(self.bottom_listbox.size()):
                value = self.bottom_listbox.get(i)

                # If the value is "NaN" or an actual NaN, skip it
                if value == "NaN" or pd.isna(value):
                    continue

                # If the value hasn't been seen yet, add it to the list
                if value not in seen_values:
                    unique_values.append(value)
                    seen_values.add(value)

            # Create a mapping of value to ordinal based on the listbox order
            recode_map = {value: i for i, value in enumerate(unique_values)}

            # Apply the recode_map to the DataFrame, replacing values in the current column
            self.df[col_name] = self.df[col_name].map(recode_map)

            # Future-proofing: If the column is of object dtype, infer objects to avoid future warning
            if self.df[col_name].dtype == "object":
                self.df[col_name] = self.df[col_name].infer_objects()

            # Display final mappings for all values with formatted output
            print("\n" + "=" * 50)  # Line separator
            print(f"Final recoding map for column '{col_name}':")

            # Display final mappings for all values
            for original_value, ordinal_value in recode_map.items():
                print(f"  {original_value} -> {ordinal_value}")

            print("=" * 50 + "\n")  # Line separator

        else:
            print("Numerical columns cannot be recoded as ordinal.")

        # Reload the column after recoding
        self.load_column(self.current_column_index)

        # If the column is now numerical, display a message "Numerical column - values not editable"
        if pd.api.types.is_numeric_dtype(self.df[col_name]):
            self.show_numerical_column_message()

    def show_numerical_column_message(self):
        """Reload the column and display a message for numerical columns."""
        self.bottom_listbox.delete(0, tk.END)
        self.bottom_listbox.insert(
            tk.END, "Numerical column - values not editable"
        )
        self.bottom_listbox.config(
            state=tk.DISABLED
        )  # Disable editing for numerical columns

    def confirm_changes(self):
        """Apply the tracked changes to the DataFrame, including NaN for missing values."""

        # Apply column name changes
        if self.top_listbox_edits:
            # First, store the old column name
            old_col_name = self.df.columns[self.current_column_index]

            # Apply the column name changes to the DataFrame
            self.df.rename(columns=self.top_listbox_edits, inplace=True)

            # Update self.bottom_listbox_edits to reflect the new column name
            new_col_name = self.top_listbox_edits.get(
                old_col_name, old_col_name
            )
            if old_col_name in self.bottom_listbox_edits:
                # Transfer edits from old column name to new column name
                self.bottom_listbox_edits[new_col_name] = (
                    self.bottom_listbox_edits.pop(old_col_name)
                )

            # Print column name changes with arrow formatting
            print("\n" + "=" * 50)  # Line separator
            print("Updated column names:")
            for old_name, new_name in self.top_listbox_edits.items():
                print(f"  {old_name} -> {new_name}")
            print("=" * 50 + "\n")  # Line separator with extra spaces

            self.top_listbox_edits.clear()  # Clear the tracked changes after applying

        # Track the column you're currently working on
        col_name = self.df.columns[self.current_column_index]

        # Check if there are changes for the current column in bottom_listbox_edits
        if col_name in self.bottom_listbox_edits:
            # Apply the changes only to the current column
            changes = self.bottom_listbox_edits[col_name]
            self.df[col_name] = self.df[col_name].replace(changes)

            # Print unique value changes with arrow formatting
            print("\n" + "=" * 50)  # Line separator
            print(f"Updated unique values in column '{col_name}':")
            for old_value, new_value in changes.items():
                print(f"  {old_value} -> {new_value}")

            # Clear the tracked changes for the current column
            self.bottom_listbox_edits.pop(col_name)

        # Print the updated DataFrame to verify changes
        print("\n" + "=" * 50)  # Line separator
        print(f"Updated DataFrame for column '{col_name}':")
        print(self.df[[col_name]].head())  # Print only the current column
        print("=" * 50 + "\n")  # Line separator with extra spaces

    def on_closing(self):
        """Handle the window close event and confirm final changes."""
        # Confirm changes before closing
        self.confirm_changes()
        self.root.quit()  # End the Tkinter main loop
        self.root.destroy()  # Destroy the root window

    def get_edited_df(self):
        """Return the updated DataFrame after the UI interaction."""
        return self.df


# Function to create a DataFrame with mixed types of columns
def create_random_df(num_rows, num_cols):
    column_names = [f"Column {i+1}" for i in range(num_cols)]
    data = {}
    for i in range(num_cols):
        if i % 3 == 0:
            # Numerical column
            data[column_names[i]] = np.random.randint(1, 100, num_rows)
        elif i % 3 == 1:
            # String column
            data[column_names[i]] = np.random.choice(
                ["apple", "banana", "cherry", "pineapple"], num_rows
            )
        else:
            # Mixed column (string + number)
            data[column_names[i]] = np.random.choice(
                ["apple", 42, "banana", 99], num_rows
            )
    df = pd.DataFrame(data)
    return df
