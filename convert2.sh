#!/bin/bash
mkdir -p output-otf
for INPUT in input/*.png; do
	echo "$INPUT"
	OUTPUT="output-otf/$(basename $INPUT .png).bbc"
	if [ ! -f "$OUTPUT" ]; then
		python mode1attr-otf2.py "$INPUT" "$OUTPUT"
	fi
done
