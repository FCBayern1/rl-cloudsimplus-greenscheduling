#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
import sys

# Get commit hash from command line
commit_hash = sys.argv[1] if len(sys.argv) > 1 else "87c553f"

# Get original message
result = subprocess.run(
    ["git", "log", "--format=%B", "-1", commit_hash],
    capture_output=True,
    text=True,
    encoding='utf-8'
)

original_msg = result.stdout

# Clean the message
lines = original_msg.split('\n')
cleaned_lines = []

for line in lines:
    # Skip Claude-related lines
    if '🤖' in line or 'Generated with' in line:
        continue
    if 'Co-Authored-By: Claude' in line:
        continue
    cleaned_lines.append(line)

# Remove trailing empty lines
while cleaned_lines and not cleaned_lines[-1].strip():
    cleaned_lines.pop()

cleaned_msg = '\n'.join(cleaned_lines)

# Output the cleaned message
print(cleaned_msg)
