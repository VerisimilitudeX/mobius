import pandas as pd
import os
import sys
import time
from tqdm import tqdm

# File paths
big_csv = "/Volumes/T9/EpiMECoV/processed_data/filtered_biomarker_matrix.csv"
out_csv = "/Users/verisimilitude/Downloads/output_small.csv"

def get_file_size(file_path):
    """Get file size in MB"""
    return os.path.getsize(file_path) / (1024 * 1024)

class LineCounter:
    """Count lines in a file with progress updates"""
    def __init__(self, filename):
        self.filename = filename
        self.file_size = get_file_size(filename)
        
    def count_lines(self):
        """Count lines in a file with progress updates"""
        print(f"Estimating total lines in file ({self.file_size:.2f} MB)...")
        line_count = 0
        
        # Create a progress bar for line counting
        with tqdm(total=100, desc="Scanning file", unit="%") as pbar:
            with open(self.filename, 'rb') as f:
                # Use binary mode for better performance
                last_percent = 0
                
                while True:
                    # Read 1MB at a time
                    buffer = f.read(1024 * 1024)
                    if not buffer:
                        break
                    
                    # Count newlines in this buffer
                    line_count += buffer.count(b'\n')
                    
                    # Update progress
                    current_pos = f.tell() / (1024 * 1024)
                    percent_complete = min(99, int(current_pos / self.file_size * 100))
                    
                    if percent_complete > last_percent:
                        pbar.update(percent_complete - last_percent)
                        last_percent = percent_complete
                        
        # Add 1 for the last line if file doesn't end with newline
        return line_count + 1

class CSVProcessor:
    def __init__(self, input_file, output_file, small_chunksize=10000):
        self.input_file = input_file
        self.output_file = output_file
        self.file_size = get_file_size(input_file)
        self.chunksize = small_chunksize  # Smaller chunks for more frequent updates
        
    def process_with_live_updates(self):
        """Process CSV with live updates to show progress"""
        print(f"File size: {self.file_size:.2f} MB")
        
        # Get an estimate of total lines for better progress tracking
        counter = LineCounter(self.input_file)
        estimated_lines = counter.count_lines()
        print(f"Estimated total lines: {estimated_lines:,}")
        
        # Track header separately
        header = None
        
        # Track if we've processed at least the first chunk
        first_chunk_processed = False
        
        # Process in chunks with progress updates
        chunks_processed = 0
        rows_processed = 0
        start_time = time.time()
        
        print("\nReading CSV data with progress monitoring...")
        with tqdm(total=estimated_lines, desc="Reading rows", unit="rows") as pbar:
            # Initial progress message
            print("Starting to read the first chunk (this may take a while for large files)...")
            
            # Read the header first (just the first line)
            try:
                header_df = pd.read_csv(self.input_file, nrows=0)
                header = header_df.columns.tolist()
                print(f"Successfully read header with {len(header)} columns")
                
                # Update progress for header
                pbar.update(1)
                rows_processed += 1
            except Exception as e:
                print(f"Error reading header: {e}")
                return None
            
            # Process rest of file in chunks
            try:
                # Initialize empty list to store processed chunks for small subset
                first_chunks = []
                
                # Create iterator for chunks
                chunk_iterator = pd.read_csv(self.input_file, chunksize=self.chunksize, 
                                           skiprows=1, header=None)
                
                # Process chunks
                for i, chunk in enumerate(chunk_iterator):
                    chunks_processed += 1
                    
                    # Assign header
                    chunk.columns = header
                    
                    # Keep track of the first chunks for small subset
                    if len(first_chunks) < 1 and rows_processed < 5:
                        first_chunks.append(chunk.head(5 - rows_processed))
                    
                    # Update progress
                    chunk_rows = len(chunk)
                    rows_processed += chunk_rows
                    pbar.update(chunk_rows)
                    
                    # Print live status every few chunks
                    if chunks_processed % 5 == 0:
                        elapsed = time.time() - start_time
                        rate = rows_processed / elapsed if elapsed > 0 else 0
                        print(f"Status: Processed {chunks_processed} chunks, {rows_processed:,} rows "
                              f"({rate:.2f} rows/sec), elapsed time: {elapsed:.1f} sec")
                    
                    # Mark first chunk processed
                    if not first_chunk_processed:
                        print(f"First chunk successfully processed! Processing continues...")
                        first_chunk_processed = True
                
                # Create final small dataframe
                small_df = pd.concat(first_chunks) if first_chunks else None
                
                return header, small_df, rows_processed
                
            except Exception as e:
                print(f"\nError during chunk processing: {e}")
                
                # If we processed at least the first chunk, we might be able to continue
                if first_chunk_processed and first_chunks:
                    print("Attempting to create output with partial data...")
                    small_df = pd.concat(first_chunks)
                    return header, small_df, rows_processed
                else:
                    print("Failed to process even the first chunk.")
                    return None
    
    def create_small_subset(self, header, small_df):
        """Create a small subset of the data"""
        if header is None or small_df is None or small_df.empty:
            print("Not enough data processed to create a subset.")
            return
        
        try:
            # Identify the last column name (which should be "Condition")
            condition_col = header[-1]
            
            # Let's pick the first 10 feature columns, plus the Condition column
            feature_cols = header[:10]  # first 10 columns
            subset_cols = feature_cols + [condition_col]
            
            # Keep only the subset of columns
            df_small = small_df[subset_cols].head(5)
            
            # Write this small subset to a new file
            print(f"Writing subset to {self.output_file}...")
            df_small.to_csv(self.output_file, index=False)
            print(f"Created a small CSV with shape {df_small.shape} at {self.output_file}")
            
        except Exception as e:
            print(f"Error creating subset: {e}")

# Main execution
try:
    print(f"Starting CSV processing with incremental progress feedback...")
    
    # Create processor
    processor = CSVProcessor(big_csv, out_csv, small_chunksize=5000)
    
    # Process file with live updates
    result = processor.process_with_live_updates()
    
    if result:
        header, small_df, rows_processed = result
        processor.create_small_subset(header, small_df)
        print(f"Processing completed successfully. Total rows processed: {rows_processed:,}")
    else:
        print("Processing failed.")
    
except KeyboardInterrupt:
    print("\nOperation cancelled by user.")
except Exception as e:
    print(f"Unexpected error: {e}")