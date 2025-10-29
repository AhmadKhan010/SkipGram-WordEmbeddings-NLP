import os
import json
import glob
from datetime import datetime

def merge_json_files(input_pattern="judgments_with_ocr_sentences_*.json",
                     output_file="merged_judgments.jsonl"):
    """
    Merges multiple JSON files into a single JSONL file.
    Skips entries where 'Download' == 'N/A'.
    Assigns sequential SrNo starting from 1 for all merged entries.
    """

    merged_records = []
    total_files = 0
    skipped_count = 0

    # Get all matching JSON files
    files = sorted(glob.glob(input_pattern))
    if not files:
        print("❌ No matching files found.")
        return

    print(f"🗂 Found {len(files)} files to merge...\n")

    # Collect all valid entries first
    for file in files:
        total_files += 1
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Handle if data is stored under a key (like data["Judgments"])
            if isinstance(data, dict) and "Judgments" in data:
                data = data["Judgments"]

            for entry in data:
                if entry.get("Download", "").strip().upper() == "N/A":
                    skipped_count += 1
                    continue

                # Add file origin and timestamp metadata
                entry["SourceFile"] = os.path.basename(file)
                entry["MergedTimestamp"] = datetime.now().isoformat()

                merged_records.append(entry)

        except Exception as e:
            print(f"⚠️ Error reading {file}: {e}")

    # Assign sequential SrNo
    for i, record in enumerate(merged_records, start=1):
        record["SrNo"] = i

    # Write final JSONL file
    with open(output_file, "w", encoding="utf-8") as outfile:
        for record in merged_records:
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Summary
    print(f"✅ Merge completed successfully!\n")
    print(f"📁 Total files processed: {total_files}")
    print(f"✔️ Total records merged: {len(merged_records)}")
    print(f"🚫 Records skipped (Download='N/A'): {skipped_count}")
    print(f"💾 Output file saved as: {output_file}")

if __name__ == "__main__":
    merge_json_files()
