# HackMerced8

Note: ($) signifies in command prompt/bash

### Steps for Training Weather Model:
$python train_ARIMA.py

  Go to weather_predictions.csv and add a line at the beginning with the name 'data'

### Steps for Training and Running Yield Prediction Model:
$python train_randomForest.py 

$python test_randomForest.py 



## Notes:
First we train the two AR models to predict future weather trends. Second, we add a line to weather_predictions.csv so that train_randomForest.py can read it. Third, we run train_randomForest.py and train a randomForest that can predict the weather.


