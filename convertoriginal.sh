#!/bin/bash
mkdir -p output-original
for INPUT in input/*.png; do
	echo "$INPUT"
	OUTPUT="output-original/$(basename $INPUT .png).png"
	if [ ! -f "$OUTPUT" ]; then
		python tweak-4bit.py "$INPUT" "$OUTPUT"
	fi
done
