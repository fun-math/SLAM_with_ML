# SLAM_with_ML

This project is an attempt to recreate the results of [HFnet](https://arxiv.org/pdf/1812.03506.pdf). We have used Superpoint model as teacher model for keypoint detector and local descriptor and NetVLAD for global descriptor. 

Run this to download weights and keep it in the weights folder

`$ wget https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.75_224_no_top.h5`

## Weights

- [Mobilenet_0.75_224_no_top](https://drive.google.com/file/d/1eNQ4c1c-sRHs8gjw_T1X9f4HgW783YQW/view?usp=sharing)
- [Super point weight](https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/superpoint_v1.pth)
