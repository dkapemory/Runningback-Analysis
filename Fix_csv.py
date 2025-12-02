import pandas as pd
import tkinter as tk
from tkinter import messagebox

def main():
    path = "condensed_train_clean.csv"
    print(f"Using fixed CSV file: {path}")

    # Load the CSV into a DataFrame
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print("Error reading CSV:", e)
        return

    # --- Build GUI ---
    root = tk.Tk()
    root.title("Delete Columns from condensed_train_clean.csv")

    # Instructions label
    label = tk.Label(
        root,
        text="Select the columns you want to delete, then click 'Delete Selected Columns'."
    )
    label.pack(padx=10, pady=10)

    # --- Scrollable frame for checkbuttons ---
    container = tk.Frame(root)
    container.pack(fill="both", expand=True, padx=10, pady=5)

    canvas = tk.Canvas(container)
    canvas.pack(side="left", fill="both", expand=True)

    scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
    scrollbar.pack(side="right", fill="y")

    canvas.configure(yscrollcommand=scrollbar.set)

    # Frame inside the canvas
    frame = tk.Frame(canvas)
    frame_id = canvas.create_window((0, 0), window=frame, anchor="nw")

    def resize_scrollregion(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    frame.bind("<Configure>", resize_scrollregion)

    # Dictionary to hold column name -> BooleanVar
    check_vars = {}

    for col in df.columns:
        var = tk.BooleanVar(value=False)
        chk = tk.Checkbutton(frame, text=col, variable=var, anchor="w", justify="left")
        chk.pack(anchor="w")
        check_vars[col] = var

    def delete_selected():
        # Get the list of columns that are checked
        cols_to_delete = [col for col, var in check_vars.items() if var.get()]

        if not cols_to_delete:
            messagebox.showinfo("No Selection", "No columns selected to delete.")
            return

        # Confirm with the user
        col_list_str = ", ".join(cols_to_delete)
        if not messagebox.askyesno(
            "Confirm Deletion",
            f"Are you sure you want to delete these columns?\n\n{col_list_str}"
        ):
            return

        try:
            new_df = df.drop(columns=cols_to_delete)
            new_df.to_csv(path, index=False)
            messagebox.showinfo(
                "Success",
                f"Deleted {len(cols_to_delete)} column(s):\n{col_list_str}\n\nSaved to {path}."
            )
            root.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Error deleting/saving:\n{e}")

    # Delete button
    delete_button = tk.Button(root, text="Delete Selected Columns", command=delete_selected)
    delete_button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
