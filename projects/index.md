---
layout: project
title: All Projects
excerpt: "A List of Projects"
---


# Yang WANG 
Data science portfolio

# [Project 1: Fusion of algorithms for face recognition](https://wangyangparis.github.io/DataChallenge2020/) 
* Data Challenge IDEMIA-MS Big Data (ranked 1st among 57 data scientists)
* build a fusion of algorithms in order to construct the best suited solution for comparison of a pair of images
* total 9,800,713 training observations. There are in total 3,768,311 test observations.
* the performance criterion is TPR for the value of FPR = 0.0001
* LightGBoost
* Custom Loss Function - In order to penalise False Positive, I put a penalty Beta on the FP, calculating the gradient as:
grad =∂𝐿/∂𝑥=∂𝐿/∂𝑝*∂𝑝/∂𝑥=𝑝(𝛽+𝑦−𝛽𝑦)−𝑦 ,
and hessien as:
hess =∂2/𝐿∂𝑥2=𝑝(1−𝑝)(𝛽+𝑦−𝛽𝑦)

<p align="center">
  <img src="https://www.statworx.com/wp-content/uploads/machine.png"  width="250" />
</p>


# [Project 2: Airbus Anomaly Detection Project](https://wangyangparis.github.io/AirbusAnomalyDetectionProject/) 
- Interpolation - Spline
- Frequency Domain - Fast Fourier Transformation
- OneClassSVM / Isolation Forest /Local Outlier Factor + PCA/Kernel PCA
- VAE latent space visualization
<p align="left"><img src="https://raw.githubusercontent.com/wangyangparis/AirbusAnomalyDetectionProject/master/Images/spectrum.png" width="350" ></img></p>

# [Project 3: Paris Wifi Data Visualization](https://wangyangparis.github.io/ParisWifiDataviz/) 
- Spotly/Dash
- Time varying data visualization
<p align="left"><img src="" width="550" ></img></p>

# [ Project: device-geolocalisation](https://wangyangparis.github.io/IoT-geolocation/) 
- A company (Sigfox) that produces connected device needs to locate the position of their device. 
- Prediction is based on the message reception information.
<p align="left"><img src="https://www.google.com/url?sa=i&url=https%3A%2F%2Ffr.wikipedia.org%2Fwiki%2FSigfox&psig=AOvVaw24F_MfqMEfWMOA9-6Yyt7H&ust=1595095559357000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCPCCu5bw1OoCFQAAAAAdAAAAABAD" width="250" ></img></p>


# [ Project: Scala Spark ML Project](https://github.com/wangyangparis/spark_project_kickstarter_2019_2020) 
- Kickstarter campaigns - text classification
- Scala RDD 
- create a ML Pipeline of 10 stages
- hyperparamater tuning with grid search
<p align="left"><img src="https://upload.wikimedia.org/wikipedia/commons/f/f3/Apache_Spark_logo.svg" width="200" ></img></p>

# [ Web Game Wikipedia-Travel-to-Philosophie](https://wangyangparis.github.io/WebGame-Wikipedia-Travel-to-Philosophie) 
- Flask, html, python, CSS

@created 2020/7/17 
==================Under construction==================


# [Project 4: IBM Text Clustering](https://wangyangparis.github.io/AirbusAnomalyDetectionProject/) 
- NLP, short text clustering
- Clustering with BERT/flauBERT text embedding
- Transformer HuggingFace, Rake

<p align="center"><img src="" width="550" ></img></p>


# [ Data Engineering Project GDELT databae request](https://wangyangparis.github.io/AirbusAnomalyDetectionProject/) 
- MongoDB, Spark, AWS S3, EC2
- NoSQL





