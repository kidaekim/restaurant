# restaurant

This is based on the Kaggle competition, 'Restaurant Revenue Prediction', hosted by TFI.

https://www.kaggle.com/c/restaurant-revenue-prediction

All the revenue outliers in the train set come from the restaurants in big cities(all in Istanbul). 

I would consider setting up a seperate model for those in the big cities first. 

By training a model without these revenue outliers, prediction accuracy on the validation data is much better.

gbm seems to work pretty well here. 

My next step would be to identify/engineer features of these revenue outliers.
