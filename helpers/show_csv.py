import os
import pandas as pd
import xml.etree.ElementTree as ET

try:
    import pyreadr
    PYREADR_AVAILABLE = True
except ImportError:
    PYREADR_AVAILABLE = False

root_folder = "/Volumes/T9/EpiMECoV/" 
output_file = "summary_of_files.txt"
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, output_file)

def summarize_csv(file_path, f):
    df = pd.read_csv(file_path)
    f.write(f"  - Dimensions: {df.shape}\n")
    f.write(f"  - Headings: {list(df.columns)}\n")
    f.write(f"  - First 5 Rows:\n{df.head().to_string(index=False)}\n")

def summarize_rds(file_path, f):
    if not PYREADR_AVAILABLE:
        f.write("  - pyreadr not installed. Cannot parse RDS content.\n")
        return
    try:
        result = pyreadr.read_r(file_path)
        for key in result.keys():
            df = result[key]
            f.write(f"  - R object: {key}, shape = {df.shape}\n")
            f.write(f"    * Columns: {list(df.columns)}\n")
            if len(df) > 0:
                f.write(f"    * First 5:\n{df.head().to_string(index=False)}\n")
    except Exception as e:
        f.write(f"  - RDS Error: {e}\n")

def summarize_idat(file_path, f):
    size = os.path.getsize(file_path)
    f.write(f"  - File size: {size} bytes\n")
    with open(file_path, "rb") as idf:
        head = idf.read(8)
    f.write(f"  - First 8 bytes (hex): {head.hex()}\n")
    f.write("  - Likely an Illumina IDAT file.\n")

def summarize_xml(file_path, f):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        c_count = len(list(root))
        f.write(f"  - Root tag: {root.tag}\n")
        f.write(f"  - # of children: {c_count}\n")
    except Exception as e:
        f.write(f"  - XML Parse Error: {e}\n")

def summarize_other(file_path, f):
    size = os.path.getsize(file_path)
    f.write(f"  - File size: {size} bytes\n")
    f.write("  - (Non-code data file: short summary only.)\n")

if __name__ == "__main__":
    all_files = []
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            path = os.path.join(subdir, file)
            all_files.append(path)

    total_files = len(all_files)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Summary of Files\n" + "="*60 + "\n\n")
        for idx, fp in enumerate(all_files, start=1):
            print(f"Processing file {idx} of {total_files}: {fp}")
            f.write(f"File {idx}/{total_files}: {fp}\n")
            ext = os.path.splitext(fp)[1].lower()
            try:
                if ext == ".csv":
                    summarize_csv(fp, f)
                elif ext == ".rds":
                    summarize_rds(fp, f)
                elif ext == ".idat":
                    summarize_idat(fp, f)
                elif ext == ".xml":
                    summarize_xml(fp, f)
                else:
                    summarize_other(fp, f)
            except Exception as e:
                f.write(f"  - Exception: {e}\n")
            f.write("\n" + "-"*40 + "\n\n")

    print(f"Summary saved to: {output_path}")
    with open(output_path, "r", encoding="utf-8") as f:
        content = f.read()
        tokens_approx = len(content.split())
    print(f"Approximate tokens in {output_file}: {tokens_approx}")