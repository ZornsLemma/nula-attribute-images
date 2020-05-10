#!/bin/bash
mkdir -p output-otf
mkdir -p output-otf-simulated
for INPUT in input/*.png; do
	echo "$INPUT"
	OUTPUT="output-otf/$(basename $INPUT .png).bbc"
	OUTPUTSIM="output-otf-simulated/$(basename $INPUT .png).png"
	if [ ! -f "$OUTPUT" ]; then
		python mode1attr-otf3.py "$INPUT" "$OUTPUT" "$OUTPUTSIM"
	fi
done
