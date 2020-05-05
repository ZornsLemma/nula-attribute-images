#!/bin/bash
mkdir -p output; cd output
for x in ../input/*.png; do
	echo "$x"
	if [ ! -f "$(basename $x .png).bbc" ]; then
		python ../mode1attr.py "$x"
	fi
done
