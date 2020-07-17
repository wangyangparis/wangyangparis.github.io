---
layout: project
title: All Projects
excerpt: "A List of Projects"
---


# Yang WANG 
Data science portfolio

# [Project 1: Fusion of algorithms for face recognition](https://wangyangparis.github.io/DataChallenge2020/) 
* build a fusion of algorithms in order to construct the best suited solution for comparison of a pair of images
* total 9,800,713 training observations. There are in total 3,768,311 test observations.
* the performance criterion is TPR for the value of FPR = 0.0001
* In order to penalise False Positive, I put a penalty Beta on the FP, I can calculate the gradient as:
grad =∂𝐿/∂𝑥=∂𝐿/∂𝑝*∂𝑝/∂𝑥=𝑝(𝛽+𝑦−𝛽𝑦)−𝑦 ,
and hessien as:
hess =∂2/𝐿∂𝑥2=𝑝(1−𝑝)(𝛽+𝑦−𝛽𝑦)

<p align="center">
  <img src="https://www.statworx.com/wp-content/uploads/machine.png"  width="450" height="450"/>
</p>


# [Project 2: Airbus Anomaly Detection Project](https://wangyangparis.github.io/AirbusAnomalyDetectionProject/) 
- Interpolation - Spline
- Frequency Domain - Fast Fourier Transformation
- OneClassSVM / Isolation Forest /Local Outlier Factor + PCA/Kernel PCA
- VAE latent space visualization
<p align="center"><img src="https://raw.githubusercontent.com/wangyangparis/AirbusAnomalyDetectionProject/master/Images/spectrum.png" width="550" ></img></p>

