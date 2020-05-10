#!/bin/bash
NAME=output-static
mkdir -p $NAME
mkdir -p $NAME-simulated
for INPUT in input/*.png; do
	echo "$INPUT"
	OUTPUT="$NAME/$(basename $INPUT .png).bbc"
	OUTPUTSIM="$NAME-simulated/$(basename $INPUT .png).png"
	if [ ! -f "$OUTPUT" ]; then
		python mode1attr.py "$INPUT" "$OUTPUT" "$OUTPUTSIM"
	fi
done
