import os
import re

# File paths
input_path = os.path.join('/Users/verisimilitude/Downloads', 'output.txt')
output_path = os.path.join('/Users/verisimilitude/Downloads', 'output_small.txt')

print(f"Reading file from: {input_path}")

try:
    # Read the entire file
    with open(input_path, 'r', encoding='utf-8') as file:
        data = file.read()
    
    # Count occurrences - using regex with word boundaries to ensure we're counting whole words
    me_count = len(re.findall(r'\bME\b', data))
    lc_count = len(re.findall(r'\bLC\b', data))
    controls_count = len(re.findall(r'\bcontrols\b', data))
    
    # Print occurrence counts
    print(f"Occurrences of 'ME': {me_count}")
    print(f"Occurrences of 'LC': {lc_count}")
    print(f"Occurrences of 'controls': {controls_count}")
    
    # Split into rows
    rows = data.split('\n')
    total_rows = len(rows)
    print(f"Total rows: {total_rows}")
    
    # Calculate 25% of rows
    rows_to_keep = int(total_rows * 0.25)
    if rows_to_keep < 1:
        rows_to_keep = 1  # Keep at least one row
    print(f"Keeping first {rows_to_keep} rows (25%)")
    
    # Extract first quarter of rows
    first_quarter = rows[:rows_to_keep]
    
    # Join back into a string
    new_content = '\n'.join(first_quarter)
    
    # Count occurrences in the reduced file as well
    me_count_reduced = len(re.findall(r'\bME\b', new_content))
    lc_count_reduced = len(re.findall(r'\bLC\b', new_content))
    controls_count_reduced = len(re.findall(r'\bcontrols\b', new_content))
    
    print(f"In reduced file:")
    print(f"Occurrences of 'ME': {me_count_reduced}")
    print(f"Occurrences of 'LC': {lc_count_reduced}")
    print(f"Occurrences of 'controls': {controls_count_reduced}")
    
    # Write to new file
    with open(output_path, 'w', encoding='utf-8') as output_file:
        output_file.write(new_content)
    
    print(f"Successfully wrote {rows_to_keep} rows to: {output_path}")
    print(f"Original size: {len(data)} characters")
    print(f"New file size: {len(new_content)} characters")
    print(f"Size reduction: {((len(data) - len(new_content)) / len(data) * 100):.2f}%")
    
except Exception as e:
    print(f"Error: {str(e)}")