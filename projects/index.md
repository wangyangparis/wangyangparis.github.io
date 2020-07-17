---
layout: project
title: All Projects
excerpt: "A List of Projects"
---


# Ken_Portfolio
Example data science portfolio

# [Project 1: Fusion of algorithms for face recognition](https://wangyangparis.github.io/Data-Challenge-2020/) 
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

![](https://www.statworx.com/wp-content/uploads/machine.png)


# [Project 2: Ball Image Classifier](https://github.com/PlayingNumbers/ball_image_classifier) 
For this example project I built a ball classifier to identify balls from different sports. This could be useful for someone who is new to sports from a certain country. They could take a picture of a ball and an app could serve them some information about the history and rules of the game. This is the underlying model for building something with those capabilities. 

I was able to get the model to predict the sport of the ball with 94% accuracy after minimal tuning. For most of the cases this would meet the need of an end user of the app. To get these results I used transfer learning on a CNN trained on resnet34. This created time efficiencies and solid results. 

![](/images/matrix_results.png)
