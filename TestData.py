import pandas as pd


def print_row_by_number(row_number):
    """
    Print a specific row from the train.csv file.
    
    Args:
        row_number (int): The row number to print (1-based indexing)
    """
    try:
        # Read the CSV file
        df = pd.read_csv('train.csv')
        
        # Convert to 0-based indexing
        index = row_number - 1
        
        # Check if the row number is valid
        if index < 0 or index >= len(df):
            print(f"Error: Row number {row_number} is out of range. File has {len(df)} rows.")
            return
            
        # Print the row
        print(f"\nRow {row_number}:")
        print(df.iloc[index])
        
    except FileNotFoundError:
        print("Error: train.csv file not found.")
    except Exception as e:
        print(f"Error occurred: {str(e)}")

# Example usage:
if __name__ == "__main__":
    # Ask the user for the row number
    try:
        row_to_print = int(input("Enter the row number to print (1-based): "))
        print_row_by_number(row_to_print)
    except ValueError:
        print("Invalid input. Please enter a valid integer.")

print ("TestData.py loaded successfully.")