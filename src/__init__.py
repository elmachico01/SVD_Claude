"""
SVD & YOLO for Image Steganography
==================================
Project package for the course *Statistical and Mathematical Methods for AI*.

Modules
-------
svd_core        Singular Value Decomposition mathematics (the course theory).
steganography   Truncated-SVD secret compression + block-SVD QIM embedding.
yolo_guidance   YOLOv8 detection → embedding guidance + downstream evaluation.
metrics         PSNR / SSIM / MSE / NC / BER / capacity.
attacks         Robustness attacks (JPEG, noise, blur, rescale, …).
dataset         COCO-128 loading and the default secret image.
pipeline        End-to-end per-image hide / reveal / evaluate.
"""
__version__ = "1.0.0"
