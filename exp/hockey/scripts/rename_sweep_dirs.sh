#!/bin/bash
# Rename misleading "non-robust" directories to show they're actually robust with ncc=300
# These directories were created with type=robust (default) but without explicit ncc/nbc in name

SWEEP_DIR="exp/hockey/outputs/sweep"

# Find directories without _nbc or _ncc (these are the fake non-robust ones)
for dir in "$SWEEP_DIR"/p1_*; do
    if [[ -d "$dir" ]] && [[ ! "$dir" =~ _nbc ]] && [[ ! "$dir" =~ _ncc ]]; then
        basename=$(basename "$dir")
        
        # Insert _p2_nbc10.0_p2_ncc300.0 before _sns
        # The default was nccw=300, nbcw=10
        newname=$(echo "$basename" | sed 's/_sns/_p2_nbc10.0_p2_ncc300.0_sns/')
        
        echo "Renaming: $basename -> $newname"
        mv "$dir" "$SWEEP_DIR/$newname"
    fi
done

echo "Done!"
