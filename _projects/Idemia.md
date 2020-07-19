---
title: "Fusion of algorithms for face recognition"
excerpt: "Data Challenge IDEMIA-MS Big Data (ranked 1st among 57 data scientists) <br>
* build a fusion of algorithms in order to construct the best suited solution for comparison of a pair of images <br>
* total 9,800,713 training observations. There are in total 3,768,311 test observations. <br>
* the performance criterion is TPR for the value of FPR = 0.0001 <br>
* LightGBoost <br>
* Custom Loss Function - In order to penalise False Positive, I put a penalty Beta on the FP, calculating the gradient as:
grad =∂𝐿/∂𝑥=∂𝐿/∂𝑝*∂𝑝/∂𝑥=𝑝(𝛽+𝑦−𝛽𝑦)−𝑦 ,
and hessien as:
hess =∂2/𝐿∂𝑥2=𝑝(1−𝑝)(𝛽+𝑦−𝛽𝑦)"
git_url: "https://wangyangparis.github.io/DataChallenge2020/ "
image: "https://raw.githubusercontent.com/wangyangparis/Data-Challenge-2020/master/Images/idemia.png "
publish: true
---