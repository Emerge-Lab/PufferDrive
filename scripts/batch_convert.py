#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_file(input_path, output_dir, converter_script, file_index):
    """Worker function to process a single file and rename it sequentially."""
    # This uses :03d, meaning minimum 3 digits padded with zeros, expanding naturally after 999.
    output_name = f"map_{file_index:03d}.bin"
    output_path = output_dir / output_name
    
    # Construct the command to call the original script
    cmd = [
        sys.executable, 
        str(converter_script), 
        str(input_path), 
        str(output_path),
        "--unique-map-id", str(file_index)
    ]
    
    try:
        # Run the script. capture_output prevents terminal spam from 24 workers
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return input_path.name, output_name, True, None
    except subprocess.CalledProcessError as e:
        return input_path.name, output_name, False, e.stderr

def main():
    parser = argparse.ArgumentParser(description="Batch process binary files into sequential map_XXX.bin files.")
    parser.add_argument("--input-dir", required=True, type=str, help="Directory containing the .bin files")
    parser.add_argument("--output-dir", required=True, type=str, help="Directory to save the converted files")
    parser.add_argument("--script", default="converter.py", type=str, help="Path to your original script")
    parser.add_argument("--workers", type=int, default=24, help="Number of parallel workers (default: 24)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    converter_script = Path(args.script)

    if not input_dir.is_dir():
        sys.exit(f"Error: Input directory '{input_dir}' does not exist.")
    if not converter_script.is_file():
        sys.exit(f"Error: Converter script '{converter_script}' not found.")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning '{input_dir}' for .bin files...")
    bin_files = list(input_dir.glob("*.bin"))
    
    # Sort files to ensure deterministic indexing across different runs
    bin_files.sort()
    
    total_files = len(bin_files)
    if total_files == 0:
        sys.exit("No .bin files found in the input directory.")
        
    print(f"Found {total_files} files. Starting processing with {args.workers} workers...\n")

    start_time = time.time()
    success_count = 0
    fail_count = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Enumerate to assign a deterministic index (0 to N-1) to each file
        futures = {
            executor.submit(process_file, f, output_dir, converter_script, idx): f 
            for idx, f in enumerate(bin_files)
        }

        for i, future in enumerate(as_completed(futures), 1):
            original_name, new_name, success, error_msg = future.result()
            
            if success:
                success_count += 1
            else:
                fail_count += 1
                print(f"\n[ERROR] Failed on {original_name} -> {new_name}: {error_msg}")

            if i % 100 == 0 or i == total_files:
                elapsed = time.time() - start_time
                rate = i / elapsed
                print(f"Progress: {i}/{total_files} ({(i/total_files)*100:.1f}%) "
                      f"| Success: {success_count} | Failed: {fail_count} "
                      f"| Rate: {rate:.1f} files/sec", end="\r")

    total_time = time.time() - start_time
    print(f"\n\nDone! Processed {total_files} files in {total_time:.2f} seconds.")
    print(f"Successfully converted: {success_count}")
    print(f"Failed: {fail_count}")

if __name__ == "__main__":
    main()