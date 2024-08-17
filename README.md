# Airbnb_Score_Model
## Introduction

This project focuses on predicting review scores for Airbnb homes using machine learning and neural networks. By categorizing Airbnb review scores into four distinct groups, we aimed to create models capable of making accurate predictions based on the data. 

The dataset used is unbalanced, which presented a significant challenge throughout the project. Addressing this imbalance was crucial to improving the model's performance. Neural networks proved particularly effective in predicting lower review scores by allowing us to weight each category differently. However, this approach also introduced certain drawbacks.

For those interested in undertaking a similar project, we recommend considering a traditional regression model using the original data format for the target variable. While our project achieved good accuracy and provides a solid framework for tackling this type of analysis, it still has room for improvement in certain areas.

---

This project serves as a valuable starting point for anyone looking to explore the prediction of Airbnb review scores through machine learning techniques.
## Data Collection

We gathered our data from Kaggle, specifically from the following dataset: [Airbnb Open Data](https://www.kaggle.com/datasets/arianazmoudeh/airbnbopendata).
Below are the key libraries and packages used in this project:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns

Here’s how you can format the Summary section:

markdown
Copy code
## Summary

In this project, we successfully predicted Airbnb review scores by applying various machine learning techniques, including neural networks. By categorizing review scores into four distinct groups, we managed to address the challenges posed by an unbalanced dataset. The use of neural networks allowed us to weight each category differently, improving the model's ability to predict lower review scores, though it did come with trade-offs.

While our approach yielded good accuracy, we identified potential areas for further improvement. Future work could explore alternative models, such as traditional regression, using the original data format for the target variable. Overall, this project serves as a strong foundation for predicting review scores and offers insights into handling unbalanced datasets in machine learning.
