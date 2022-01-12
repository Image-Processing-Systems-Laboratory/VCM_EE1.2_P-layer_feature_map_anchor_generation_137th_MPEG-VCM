# VCM_EE1.2_P-layer_feature_map_anchor_generation_137th_MPEG-VCM

#######################################################

Author: Minhun Lee, Hansol Choi, Seungjin Park, Minsub Kim, and Donggyu Sim

E-mail: {minhun, hschoi95, promo, minsub20, dgsim}@kw.ac.kr

#######################################################
[Introduction]

This package contains scripts to generate anchor results of object detection on P-layer features (p2, p3, p4, p5) extracted from OpenImages dataset for MPEG Video Coding for Machines(VCM). 

Please note that this test procedure is organized based on Nokia's latest contribution(m57343) for generating VCM anchor on the OpenImages dataset V6.

#######################################################
[Software environment]

Ubuntu    20.04.1 LTS

Python    3.8.11

Torch    1.9.0

Detectron2   0.5

Object-detection   0.1

Pandas   1.3.3

Numpy   1.21.2

Opencv-python 4.5.3.56

Pillow   8.3.1

ffmpeg   4.4

VTM   12.0

#######################################################
[Faster-RCNN model parameter]

Download the Faster-RCNN model parameters from the following link: 
  https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl
  
Place the downloaded `model_final_68b088.pkl` file in the `models/x101fpn/` directory. 
  
#######################################################
[OpenImages V6 Dataset]

Download OpenImages V6 validation set according to the instruction from the following webpage:
  https://storage.googleapis.com/openimages/web/challenge2019_downloads.html

The downloaded `validation.tar.gz` file has size of 12G bytes and  contains 41620 jpg images.
Untar this file to directory dataset/validation

For annotations, we have already set files to './dataset/annotations/' and './oi_eval/' directories.
For dataset, have to move only 5k images of the OpenImages dataset V6 to './dataset/val_openimage_v6/*.jpg' directory as below.

#######################################################
[Dataset directory structure]

./dataset/val_openimage_v6/
     0a1bd356f90aaab6.jpg
     ...
     ffddf3805faf3cbf.jpg # only 5k images
 
#######################################################
[Instructions]

Please run 'demo.sh' script to generate P-layer anchor results. The outputs will be stored in './feature/' and './output/' directories which are generated automatically, and the results from our experiments are also included in 'P-layer_anchor_report.xlsm' file. This top procedure consits of three phases as below.

In the first phase, the P-layer features are extracted from the faster_rcnn_X_101_32x8d_FPN_3 network, and the extracted P-layer features are stored as YUV 4:0:0 format using FFmpeg (png to yuv) after tiling and uniform quantization (10-bits). For feature tiling into YUV 4:0:0, we arranged 256 channels of the p2, p3, p4, and p5 feature maps in a raster scanning order, respectively, so that each YUV 4:0:0 data includes 2D feature for each input image. For the uniform quantisation process, we measured the global maximum and minimum values in the P-layer features over the whole dataset, and the the global maximum and minimum values were 20.3891 and -22.3948, respectively.

In the second phase, the YUV format data are encoded and then decoded via VTM 12.0 software with six different QP values, 35, 37, 39, 41, 43 and 45. Here we store the encoded bitstreams ('./feature/{QP}_bit/') and the reconstructed YUV format data ('./feature/{QP}_rec/') and the original feature map data ('./feature/{QP}_ori/') in the designated directory for each QP value.
In addition, please note that we actually performed the encoding jobs in a parallel manner using threading, the thread is setting the default value '4', you can change the value at each './settings/{QP}.json' file.

In the thrid phase, we calculate the bit-per-pixel(bpp) and measure the mAP performance for each QP, based on the bitstreams and the reconstructions generated at the phase two. And the result files are stored './output/{QP}_AP.txt' for each qp value.

