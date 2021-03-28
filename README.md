# LSTM Crypto Currency Price Predictor
![deep_learning](deep-learning.jpg)

Due to the volatility of cryptocurrencies, investors may want to incorporate the alternative data when forecasting their prices, such as the sentiment derived from social media and news articles to help guide their trading strategies. One such indicator is the **Crypto Fear and Greed Index (FNG)** which attempts to use a variety of data sources to produce a daily FNG value for cryptocurrencies. 

In this exercise, I build and evaluate deep learning models using both the *FNG* values and simple *closing prices* to determine if the FNG indicator provides a better signal for cryptocurrencies than the normal closing price data.
I am using the *deep learning recurrent neural networks* (RNN) to model Bitcoin closing prices. One model uses the FNG indicators to predict the closing price while the second model uses a window of closing prices to predict the nth closing price.

Both models are utilising the following steps:

1. Prepare the data for training and testing
2. Build and train custom LSTM RNNs
3. Evaluate the performance of each model

# Files
There are two Jupyter Notebook files used in this exercise: 
1. lstm_stock_predictor_fng.ipynb
2. lstm_stock_predictor_closing.ipynb

There are also two date source files:
1. btc_sentiment.csv
2. btc_historic.csv

# Prepare the Data for Training and Testing
The two Jupyter Notebook files are used to build the RNN models. I use the `window_data` function to create the window of time for the data in each dataset.

For the ***Fear and Greed (FNG)** model, I use the FNG values to predict the closing prices.

For the **Closing Prices** model, I use the previous closing prices to predict the next closing price. 

Each model uses 70% of the data for training and 30% of the data for testing.

I apply a MinMaxScaler to the X and y values to scale the data for the model.

Finally, I reshape the X_train and X_test values to fit the model's requirement of samples, time steps, and features. 


# Build and train custom LSTM RNNs
In each Jupyter Notebook, I create the same custom LSTM RNN architecture. In one notebook, I fit the data using the FNG values. In the second notebook, I fit the data using only closing prices.
Both models use the same parameters and training steps so that both models could be compared to each other accurately. 

# Evaluate the performance of each model
Finally, I use the testing data to evaluate each model and compare the performance.

# Summary
Both models were initially trained with the *window size* and *epochs* parameters set at `10` and `10` respectively. I then experimented with both parameters by setting the *window size* in the range from `20` to `3` and the number of *epochs* from `10` to `50`. The final models presented here use the *window size* of `10` and the number of *epochs* at `50`. 

The two models produced very different results which are summarised below:
## Which model has a lower loss?
The **Closing Prices** model had much lower loss than that of the **FNG model**:

**Closing Prices - 0.0043**

**FNG Model - 0.0479**

What’s more, while the **Closing Prices** model trained relatively well with the *loss* falling from **0.0449** for epoch 1 to **0.0043** for epoch 50, the **FNG Model** could not be trained at all with the *loss* remaining at the same level from epoch 1 through epoch 50 (from 0.046 to 0.0479).

## Which model tracks the actual values better over time?
The **Closing Prices** model track the actual values much better than the **FNG Model** whose predicted prices show almost no link with the actual prices.


## Which window size works best for the model?
For the **Closing Prices** model, increasing the window size from the initial value of 10 to 20 or decreasing it to 5 produced a smaller loss in both cases. This is probably deu to the fact that the longer term window was able to capture some longer term monthly patterns (20 working days is one calendar month) whereas the shorter window could respond faster to short term volatility. In the end, I kept the window size at 10.


For the **FNG model**, reducing the window size to 5 and then to 3 improved significantly the link between the predicted and the actual prices. At the same time, increasing the window size to 20 produced the predicted prices which did not show any variability at all (straight line). I therefore kept the window size at 10 so that it is the same as for the **Closing Prices** model.

# Conclusion
Using the FNG index does not provide a better insight and better ability to forecast Bitcoin prices with LSTM RNN models.
