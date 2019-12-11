#!/usr/bin/env bash
echo 'Running las2txt.'

#./LAStools/bin/las2txt -i $(ls ./*.las | sort -n | head -1) -o unclassified_cloud.txt -parse xyzcRGB

SCALE_WIDTH=$1

ORIGINAL_WIDTH=$(identify -format '%w' $(ls unifiedparsing/input_images/*.JPG | sort -n | head -1))
ORIGINAL_HEIGHT=$(identify -format '%h' $(ls unifiedparsing/input_images/*.JPG | sort -n | head -1))

SCALE_FACTOR=$(echo "scale=4; ${ORIGINAL_WIDTH} / ${SCALE_WIDTH}" | bc -l)

mkdir -p unifiedparsing/resized_images
mkdir -p unifiedparsing/segmented_images

echo 'Resizing input images.'

cp unifiedparsing/input_images/*.JPG unifiedparsing/resized_images/.
find 'unifiedparsing/resized_images' -iname '*.jpg' -exec convert \{} -verbose -resize $SCALE_WIDTH\> \{} \;

echo 'Running Unified Parsing segmentation on images.'

cd $UPP_DIR

MODEL_PATH=upp-resnet50-upernet
RESULT_PATH=./

ENCODER=$MODEL_PATH/encoder_epoch_40.pth
DECODER=$MODEL_PATH/decoder_epoch_40.pth


if [ ! -e $ENCODER ]; then
  mkdir $MODEL_PATH
fi
if [ ! -e $ENCODER ]; then
  wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/unified_perceptual_parsing/$ENCODER
fi
if [ ! -e $DECODER ]; then
  wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/unified_perceptual_parsing/$DECODER
fi

python3 -u run_upp.py

cd $PROJECT_DIR

echo 'Running processing script to map segmented pixels to points.'

python3 -u process.py $ORIGINAL_WIDTH $ORIGINAL_HEIGHT $SCALE_FACTOR

echo 'Running txt2las.'

./LAStools/bin/txt2las -i classified_cloud.txt -o classified.las -parse xyzcRGB

echo 'Complete.'