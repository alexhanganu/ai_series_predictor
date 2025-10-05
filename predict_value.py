
"""
scripts trains models using time-series
and predicts future values


Best machine learning models for time series forecasting

Naïve model.
Exponential smoothing model.
ARIMA/SARIMA.
Linear regression method.
Multi-Layer Perceptron (MLP)
Recurrent Neural Network (RNN)
Long Short-Term Memory (LSTM)

Generative AI Models good for time series:
Autoregressive Models:
	Generate sequences by modeling the conditional 
	probability of each data point given the previous data points.
		ARIMA: Autoregressive Integrated Moving Average 
		SARIMA: Seasonal Autoregressive Integrated Moving-Average

RNN / Recurrent Neural Networks:
	Good for forecasting and anomaly detection.
	NN (neural networks) classs that has recurrent connections,
	allowing them to maintain a memory of previous inputs.
	* LSTM: Long Short-Term Memory Networks: is a specific type of 
		RNN that addresses the vanishing gradient problem, making it more 
		effective in capturing long-term dependencies in time series data.

GAN / Generative Adversarial Networks:
	Good for synthetic time series data, and data augmentation.
	GANs are a type of generative model that consists of two neural
	networks, a generator, and a discriminator.

VAE / Variational Autoencoders:
	Good for anomaly detection and missing data imputation.
	VAEs are another type of generative model that can learn a probabilistic
	representation of the input data.

Transformer Models:
	Good for parallel computation
	Transformers, originally developed for natural language processing,
	have shown promise in processing sequential data, 
	including time series. They can capture long-range dependencies.

"""
import os

import pandas
import numpy

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense

#==================================================================================
#==================================================================================
# MODELING

def model_make_lstm(X, y,
                    path_2save_model,
                    dl_input_shape,
                    dl_units = 50,
                    dense_units = (5,),
                    epochs = 100,
                    train_batch_size = 32,
                    save_model = False):
    """ Define and train an LSTM model
        LSTM = Long Short-Term Memory Network
    Args:
    """    
    # Define the LSTM model architecture
    model = Sequential()
    model.add(LSTM(units = dl_units,
                   input_shape = dl_input_shape,
                   activation='relu',
                   kernel_initializer='lecun_uniform'))
    for dense_unit in dense_units:
        # Output layer to predict the next number of values to predict
        model.add(Dense(units = dense_unit))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='mean_squared_error')
    
    # Train the model
    model.fit(X, y, epochs = epochs, batch_size = train_batch_size, verbose=0)
    if save_model:
        model.save(path_2save_model)
    return model


#==================================================================================
#==================================================================================
# PREPARING DATA for model training

def prepare_data_4modelling(data,
                            look_back = 10,
                            nr_vals_2predict = 5,
                            nr_splits = 30):
    """
    Args:
        data: numpy.ndarray, data for modelling
        look_back: int(), number of splits/columns to create from the dataframe
        nr_vals_2predict: number of values to predict
    Return:
        X: list(), data X used to analyze
        y: list(), data y that is predicted
    """
    X, y = [], []
    if len(data.shape) > 1:
        for i in range(len(data) - look_back - nr_vals_2predict):
            X.append(data[i:i + look_back])
            # Predict next nr_vals_2predict values of column 0
            y.append(data[i + look_back:i + look_back + nr_vals_2predict, 0])
    else:
        for i in range(nr_splits, data.shape[0]):
            X.append(data[i-nr_splits:i, 0])
            y.append(data[i, 0])
    return numpy.array(X), numpy.array(y)

#==================================================================================
#==================================================================================

def predict_tick_with_correlations(dict_with_data, look_back, nr_vals_2predict,
                                   p_models):
    """Use the trained model to make predictions on a new grid of n columns
    Args:
        dict_with_data = dictionary with pandas.DataFrame() data for each ticker
        look_back = 
        nr_vals_2predict =
    Return:
        predicted_values: list()
    """
    # Prepare new data for prediction
    column_to_predict = "Close"
    model_name = f'model.keras'
    path_2save_model = os.path.join(p_models, model_name)
    data = pandas.DataFrame()
    for ticker in dict_with_data:
        data[f"{ticker}_{column_to_predict}"] = dict_with_data[ticker][column_to_predict]

    data_np = data.values
    X, y = prepare_data_4modelling(data_np,
                                   look_back,
                                   nr_vals_2predict)
    # Reshape input data to match LSTM input shape
    X = numpy.reshape(X, (X.shape[0], X.shape[1], X.shape[-1]))  # Assuming n columns
    model = model_make_lstm(X, y,
                           path_2save_model,
                           dl_input_shape = (look_back, X.shape[-1]),
                           dl_units = 50,
                           dense_units = (nr_vals_2predict,),
                           epochs = 100,
                           train_batch_size = 32)

    X_new, _ = prepare_data_4modelling(data_np,
                                         look_back,
                                         nr_vals_2predict)

    # Reshape input data to match LSTM input shape
    X_new = numpy.reshape(X_new, (X_new.shape[0],
                                  X_new.shape[1],
                                  data_np.shape[1]))  # Assuming n columns

    # Make predictions using the trained model
    predicted_values = model.predict(X_new)
    print(f"    NEXT predicted value for: {list(dict_with_data.keys())[0]} is: {predicted_values[-1:]}")

