# Portfolio

On this page I demosntrate some of the projects I have been working on. All the projects are related to Data Science, Machine Learning and different AI topics.

---
## Image Denoising using Convolutional Autoencoders

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/GioFr/CAE)

In this notebook I've used Convolutional Autoencoders to denoise MNIST image data, as well as deployed simple CNN for classification.
Firstly the noise has been added to the image data and afterwards it has been reconstructed by using CAE. 

CNN achieved the accuracy of 0,99.

After applying CAE to noisy MNIST data we get well recovered/denoised images,which is shown below
![](/images/Denoised.png)

---
## Counting Fingers using Convex Hull

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/GioFr/Counting_Fingers)


In this notebook we are using Convex Hull algorithm from OpenCv to count fingers. 

First of all we choose the region of interest which will be used as a background and where the hand will be positioned. We threshold it so we can grab a foreground. Afterwards when the hand is positioned in the roi, we segment it using convex hull. It looks something like the following

<p align="center">
  <img src="images/hand_convex.png" />
</p>

After that we find the maximum euclidean distance between the center of the palm and the most extreme points of the convex hull, later it will be used to create a circle with 80% radius of the max euclidean distance between center point and outermost points. And finally we loop through the contours to see if we count any more fingers.

The results are given below. Altough it works fine, the method is not robust to noise.

<img src='images/1.png' width='160'> <img src='images/2.png' width='160'> <img src='images/3.png' width='160'> <img src='images/4.png' width='160'> <img src='images/5.png' width='160'>

## Kaggle: Titanic - Classification Problem

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/GioFr/Titanic)

In this notebook we perform some data analysis on Titanic dataset provided on Kaggle. 

The objective of this task is to predict the survival rate of passengers. It's clearly a classification probem, thus we can use several machine learning or deep learning techniques to tackle this problem. In this case I am using typical machine learning technics such as: Support Vector Machines, KNN, Logistic Regression, Random Forest, Naive Bayes, Linear SVC, Decision Tree. Before applying the ML techniques to data first it is necessary to perform some data preprocessing to unfold some correlations between different features, including some feature removal or creating some additional more descriptive features. 

By plotting data we can check some dependencies that is neccasarry to know in order to evaluate which features we need to prioritize and which ones to ignore. Below for example is given the clear dependancy between age and survivale rate

![](images/age.png)

By uncovering such dependencies we can transform data, for example in case of age feature instead of continous data we can select the age range that is more or less likely to survive. 

After the preprocessing now the above mentioned machine learning techniques are applied to predict the results. Accuracy scores of each technique is given below

![](images/results.png)

## Kaggle: USA_Housing - Regression Problem

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/GioFr/Pricing)

In this notebook we perform some data analysis on USA_Housing dataset provided on Kaggle. This is clearly a regression problem where we are supposed to predict the cantinuous value of the house price. 

Before applying any machine learning technique, first data is analysed and preprossed. In the data there is no missing value which makes it so much easier to preprocess the data. In addition to that there are only 6 independent variable among which is an address feature. The pnly useful information that we can subtract from the address feature is the State. 

The best way to quickly evaluate some dependencies between features is to plot the heatmap that show some correlations between data.

![](images/heatmap.png)

Finally several machine learning techniques have been tested to predict the prices. The performance is evaluated by using R-squared and the results are given below.

![](images/priceresults.png)
