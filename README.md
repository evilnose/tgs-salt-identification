# Kaggle - TGS Salt Identification Challenge
[challenge link](https://www.kaggle.com/c/tgs-salt-identification-challenge)

Prototyped using Keras.

The challenge is done. The UNet implementation that I borrowed from [zhixuhao](https://github.com/zhixuhao/unet) 
performed best, but I also tried and tuned a few of my own implementations on GCP. UNet reaches an IOU of above 60%,
while my own attempt at predicting the bounding box first yields an IOU of around 40%. It is unknown whether
my network could reach an IOU of even higher than 40% (unlikely), but it doesn't really matter, as this
is mainly a practice for building and tuning CNN's and submitting training jobs to GCP ML Engine.

My networks are trimmed-down and modified versions of different versions of YOLO. You can find them in
trainer/models.py. The network that performed best (aside from UNet) was YoloReduced.

I also have a couple of scripts for submitting training jobs to GCP in the scripts/ folder.
