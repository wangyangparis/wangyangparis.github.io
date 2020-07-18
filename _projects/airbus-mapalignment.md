---
title: "Fusion of algorithms for face recognition"
excerpt: "Data Challenge IDEMIA-MS Big Data (ranked 1st among 57 data scientists)
* build a fusion of algorithms in order to construct the best suited solution for comparison of a pair of images
* total 9,800,713 training observations. There are in total 3,768,311 test observations.
* the performance criterion is TPR for the value of FPR = 0.0001
* LightGBoost
* Custom Loss Function - In order to penalise False Positive, I put a penalty Beta on the FP, calculating the gradient as:
grad =∂𝐿/∂𝑥=∂𝐿/∂𝑝*∂𝑝/∂𝑥=𝑝(𝛽+𝑦−𝛽𝑦)−𝑦 ,
and hessien as:
hess =∂2/𝐿∂𝑥2=𝑝(1−𝑝)(𝛽+𝑦−𝛽𝑦)"
git_url: "https://wangyangparis.github.io/DataChallenge2020/"
image: "https://upload.wikimedia.org/wikipedia/commons/2/2e/IDEMIA_Logo.jpg"
publish: true
---