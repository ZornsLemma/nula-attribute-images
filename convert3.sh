#!/bin/bash
NAME=output-otf3
mkdir -p $NAME
mkdir -p $NAME-simulated
for INPUT in input/*16c.png; do
	echo "$INPUT"
	OUTPUT="$NAME/$(basename $INPUT .png).bbc"
	OUTPUTSIM="$NAME-simulated/$(basename $INPUT .png).png"
	if [ ! -f "$OUTPUT" ]; then
		python mode1attr-otf3.py "$INPUT" "$OUTPUT" "$OUTPUTSIM"
	fi
done
