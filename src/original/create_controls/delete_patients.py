import os

raw_data_dir = "/Volumes/T9/EpiMECoV/data/controls/GSE42861/GSE42861_RAW"
gsm_file = "/Volumes/T9/EpiMECoV/src/create_controls/control_gsms.txt"

with open(gsm_file, "r") as f:
    control_gsms = set(line.strip() for line in f if line.strip())

deleted_files_count = 0
retained_files_count = 0
unmatched_gsm_ids = []

for file_name in os.listdir(raw_data_dir):
    file_path = os.path.join(raw_data_dir, file_name)
    if file_name.startswith(".") or os.path.isdir(file_path):
        continue
    gsm_id = file_name.split("_")[0].strip()
    if gsm_id not in control_gsms:
        unmatched_gsm_ids.append(gsm_id)
        try:
            os.remove(file_path)
            deleted_files_count += 1
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    else:
        retained_files_count += 1

print(f"\nCleanup done. Deleted={deleted_files_count}, Retained={retained_files_count}.")
if unmatched_gsm_ids:
    print("Sample of unmatched GSM IDs:", unmatched_gsm_ids[:10])