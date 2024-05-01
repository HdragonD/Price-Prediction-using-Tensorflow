# Price-Prediction-using-Tensorflow
Machine learning project using TensorFlow

Data Preprocessing

The code starts by installing the necessary libraries, including TensorFlow, Pandas, and Matplotlib.
It then reads in a CSV file containing stock data, likely historical stock prices and related features.
The data is preprocessed by converting the 'date' column to a datetime format and scaling the 'close' prices using the MinMaxScaler from scikit-learn.
The scaled data is then split into training and testing sets, with the training set comprising 95% of the data.
The training data is further processed by creating input features (x_train) and labels (y_train) using a sliding window approach. This is a common technique for preparing time series data for sequence prediction tasks.

Model Architecture

The code defines a Sequential model in Keras, which is a popular deep learning library.
The model consists of two LSTM (Long Short-Term Memory) layers, each with 64 units. LSTM is a type of recurrent neural network (RNN) that is well-suited for handling sequential data, such as time series.
After the LSTM layers, the model has a Dense layer with 32 units, followed by a Dropout layer with a rate of 0.5 to prevent overfitting.
The final layer is a Dense layer with a single unit, which is used for the regression task of predicting stock prices.

Model Training and Evaluation

Compiling the model with appropriate loss function and optimizer.
Training the model on the x_train and y_train data.
Evaluating the model's performance on the test set (x_test and y_test).
Potentially fine-tuning the model hyperparameters or architecture based on the evaluation results.
