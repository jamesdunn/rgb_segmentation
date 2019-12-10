#!/usr/bin/env bash

./LAStools/bin/las2txt -i $(ls ./*.las | sort -n | head -1) -o unclassified_cloud.txt -parse xyzcRGB

SCALE_WIDTH=$0
SCALE_HEIGHT=$1

ORIGINAL_WIDTH=$(identify -format '%w' $(ls input_images/*.JPG | sort -n | head -1))
ORIGINAL_HEIGHT=$(identify -format '%h' $(ls input_images/*.JPG | sort -n | head -1))

SCALE_FACTOR=$(echo "$ORIGINAL_WIDTH/$SCALE_WIDTH" | bc)

mkdir resized_images
cp input_images/*.jpg resized_images/.
find './resized_images' -iname '*.jpg' -exec convert \{} -verbose -resize $WIDTHx$HEIGHT\> \{} \;

python3 -u run_upp.py

python3 -u process.py $ORIGINAL_WIDTH $ORIGINAL_HEIGHT $SCALE_FACTOR

./LAStools/bin/txt2las -i classified_cloud.txt -o classified.las -parse xyzcRGB
