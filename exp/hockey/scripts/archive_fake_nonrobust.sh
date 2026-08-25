#!/bin/bash
# Archive misleading "non-robust" directories that are actually robust with default (nccw=300, nbcw=10)
# These should be kept separate from the real sweep data

SWEEP_DIR="exp/hockey/outputs/sweep"
ARCHIVE_DIR="exp/hockey/outputs/sweep_archive_ncc300_nbcw10"

# Create archive directory
mkdir -p "$ARCHIVE_DIR"

count=0

# Find directories without _nbc or _ncc (these are the fake non-robust ones)
for dir in "$SWEEP_DIR"/p1_*; do
    if [[ -d "$dir" ]] && [[ ! "$dir" =~ _nbc ]] && [[ ! "$dir" =~ _ncc ]]; then
        basename=$(basename "$dir")
        echo "Moving: $basename"
        mv "$dir" "$ARCHIVE_DIR/$basename"
        ((count++))
    fi
done

echo ""
echo "Moved $count directories to $ARCHIVE_DIR"
echo "These contain robust experiments with default nccw=300, nbcw=10"
