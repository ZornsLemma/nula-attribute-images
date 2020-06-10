#!/bin/bash

function doifnotexist() {
	INPUT="$1"
	OUTPUT="$NAME/$(basename $INPUT .png).bbc"
	OUTPUTSIM="$NAME-simulated/$(basename $INPUT .png).png"
	echo "$INPUT"
	if [ ! -f "$OUTPUT" ]; then
		shift
		python mode1attr-otfc2.py "$@" --simulated-output="$OUTPUTSIM" "$INPUT" "$OUTPUT" 
		exomizer raw -P0 -c -m 512 "$OUTPUT" -o "$OUTPUT.exo"
	fi
}

NAME=output-otfc
mkdir -p $NAME
mkdir -p $NAME-simulated

for INPUT in input/*32c.png; do
	doifnotexist "$INPUT"
done
doifnotexist input/CIMG0900-240x256-48c.png --max-aux-changes=3 --serpentine
doifnotexist input/CIMG05465-240x256-48c.png --max-aux-changes=3
doifnotexist input/IMG_20180609_1900447-cropped2-240x256-64c.png --max-aux-changes=4
doifnotexist input/IMG_20180720_1301471-cropped2-240x256-64c.png --max-aux-changes=2
doifnotexist input/CIMG05573-240x256-64c.png --max-aux-changes=4
doifnotexist input/parrot3-240x256-64c.png --max-aux-changes=3
doifnotexist input/CIMG09707-saturated2-240x256-64c.png --max-aux-changes=3
doifnotexist input/IMG_08419-240x256-64c.png --max-aux-changes=4
doifnotexist input/IMG_0821-240x256-64c.png --max-aux-changes=3
doifnotexist input/IMG_11526-240x256-64c.png --max-aux-changes=2
doifnotexist input/CIMG05640-240x256-64c.png --max-aux-changes=4
