#### HOG(Histogram of Oriented Gradients): An Overview

Histogram of Oriented Gradients, also known as HOG, is a feature descriptor like the Canny Edge Detector, SIFT(Scale Invariant and Feature Transform). The technique counts occurrences of gradient orientation in the localized portion of an image. The HOG descriptor focuses on the structure or the shape of an object.

**It is better than any edge descriptor as it uses magnitude as well as angle of the gradient to compute the features.**

##### Compute HOG

1. compute the gradient of the image. 
   
   $$
   G_x(r,c)=I(r,c+1)-I(r,c-1)G_y(r,c)=I(r-1,c)-I(r+1, c)
   $$
   
   After calculating $G_x$and$G_y$，magnitude and angle of each pixel is calculated using the formulate
   
   mentioned below
   
   $$
   Magnitude(\mu)=\sqrt{G_x^2+G_y^2};Angle(\theta)=|tan^{-1}(G_y/G_x)|
   $$

2. After obtaining the gradient of each pixel, the gradient matrices(magnitude and angle matrix) are divided into 8*8 cells to form a block. For each block, a 9-point histogram is calculated. A 9-point histogram develops a histogram with 9 bins and each bin has an angle range of 20 degrees.

reference: https://zhuanlan.zhihu.com/p/85829145