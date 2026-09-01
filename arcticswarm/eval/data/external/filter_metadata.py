#!/usr/bin/env python3
# Copyright 2026 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Filter metadata.jsonl to only include entries with IDs in subset_ids.json"""

import json
from pathlib import Path

def main():
    # Read the subset IDs
    subset_ids_path = Path(__file__).parent.parent / "arcticswarm" / "eval" / "data" / "subset_ids.json"
    with open(subset_ids_path, 'r') as f:
        subset_ids = set(json.load(f))

    print(f"Loaded {len(subset_ids)} subset IDs")

    # Filter metadata.jsonl
    metadata_path = Path(__file__).parent.parent / "arcticswarm" / "eval" / "data" / "metadata.jsonl"
    output_path = Path(__file__).parent.parent / "arcticswarm" / "eval" / "data" / "metadata_filtered.jsonl"

    matched_count = 0
    total_count = 0

    with open(metadata_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            total_count += 1
            try:
                entry = json.loads(line.strip())
                # Check if the task_id is in the subset
                task_id = entry.get('task_id')
                if task_id in subset_ids:
                    outfile.write(line)
                    matched_count += 1
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {total_count}: {e}")

    print(f"Processed {total_count} entries")
    print(f"Matched {matched_count} entries")
    print(f"Filtered metadata written to: {output_path}")

if __name__ == "__main__":
    main()
