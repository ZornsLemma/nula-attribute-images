#!/bin/bash
mkdir -p output
for INPUT in input/*.png; do
	echo "$INPUT"
	OUTPUT="output/$(basename $INPUT .png).bbc"
	if [ ! -f "$OUTPUT" ]; then
		python mode1attr.py "$INPUT" "$OUTPUT"
	fi
done
