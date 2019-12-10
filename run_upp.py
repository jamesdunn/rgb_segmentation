#!/usr/bin/python3
import os
import subprocess

def main():
    model_path = "upp-resnet50-upernet"
    result_path = "./segmented_images/"
    segm_downsample_rate = 8
    gpu_id = 0
    padding_constant = 8
    encoder = model_path + "/encoder_epoch_40.pth"
    decoder = model_path + "/decoder_epoch_40.pth"

    for filename in os.listdir("./resized_images") :
        os.system("python3 ./unifiedparsing/test.py" \
                    + " --model_path " + str(model_path) \
                    + " --test_img " + "./resized_images/" + str(filename) \
                    + " --arch_encoder resnet50" \
                    + " --arch_decoder upernet" \
                    + " --result " + str(result_path) \
                    + " --gpu_id " + str(gpu_id) \
                    + " --segm_downsampling_rate " + str(segm_downsample_rate) \
                    + " --padding_constant " + str(padding_constant)
                    )

if __name__ == "__main__":
    main()