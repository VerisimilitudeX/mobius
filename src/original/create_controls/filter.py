import re

file_in = "/Users/verisimilitude/Downloads/GSE42861_series_matrix.txt"
file_out = "/Volumes/T9/EpiMECoV/control_gsms.txt"

with open(file_in,"r") as fin:
    lines = fin.readlines()

control_gsms = []
for i, line in enumerate(lines):
    if "Normal genomic DNA" in line:
        gsm_matches = re.findall(r"GSM\d+", line)
        if not gsm_matches:
            next_line = lines[i+1] if i+1<len(lines) else ""
            gsm_matches = re.findall(r"GSM\d+", next_line)
        control_gsms.extend(gsm_matches)

with open(file_out,"w") as fout:
    fout.write("\n".join(control_gsms))

print("Control GSM IDs =>", len(control_gsms))