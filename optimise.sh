#!/bin/bash

if [ "$#" != "1" ]; then
    echo Syntax: $0 source-image > /dev/stderr
    exit 1
fi

OUTDIR="output-optimise"
mkdir -p "$OUTDIR"
BASEOUTNAME="$OUTDIR/$(basename $1 .png)"
# TODO: Add other tweakable parameters
for SERPENTINE in "" "--serpentine"; do
    for MAXAUX in 2 3 4; do
        OUTNAME="${BASEOUTNAME}${SERPENTINE}--max-aux-changes-$MAXAUX"
        python mode1attr-otfc2.py $SERPENTINE --max-aux-changes=$MAXAUX --simulated-output="$OUTNAME.png" "$1" "$OUTNAME.bbc" > "$OUTNAME.txt"
    done
done
