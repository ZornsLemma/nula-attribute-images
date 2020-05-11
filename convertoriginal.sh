#!/bin/bash
mkdir -p output-original
for INPUT in input/*.png; do
	echo "$INPUT"
	OUTPUT="output-original/$(basename $INPUT .png).png"
	if [ ! -f "$OUTPUT" ]; then
		convert "$INPUT" -sample 1280x1024\! "$OUTPUT"
	fi
done
