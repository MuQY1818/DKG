import pandas as pd
import os

def convert_xlsx_to_csv(xlsx_path, csv_path, sheet_name=0):
    """
    Converts a specific sheet from an XLSX file to a CSV file.

    Args:
        xlsx_path (str): The path to the input XLSX file.
        csv_path (str): The path to the output CSV file.
        sheet_name (str or int, optional): The name or index of the sheet to convert. 
                                           Defaults to 0 (the first sheet).
    """
    try:
        print(f"Reading '{os.path.basename(xlsx_path)}'...")
        # It's important to have 'openpyxl' installed for pandas to read xlsx files.
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
        
        print(f"Writing to '{os.path.basename(csv_path)}'...")
        df.to_csv(csv_path, index=False)
        
        print(f"Successfully converted '{os.path.basename(xlsx_path)}' to '{os.path.basename(csv_path)}'.\n")
    except FileNotFoundError:
        print(f"Error: Input file not found at '{xlsx_path}'\n")
    except Exception as e:
        print(f"An error occurred while converting '{os.path.basename(xlsx_path)}': {e}\n")

if __name__ == "__main__":
    # This script assumes it's run from the project root '揭榜挂帅'.
    dataset_dir = 'dataset'
    
    files_to_convert = {
        'assistments_2009_2010.xlsx': 'assistments_2009_2010.csv',
        'skill_builder_data09-10.xlsx': 'skill_builder_data09-10.csv'
    }
    
    print("Starting XLSX to CSV conversion...\n")
    
    for xlsx_file, csv_file in files_to_convert.items():
        xlsx_full_path = os.path.join(dataset_dir, xlsx_file)
        csv_full_path = os.path.join(dataset_dir, csv_file)
        
        convert_xlsx_to_csv(xlsx_full_path, csv_full_path)
        
    print("Conversion process finished.") 