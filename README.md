# Recurrent Neural Network (RNN) for Reuters News Classification

## Project Overview
This notebook demonstrates the implementation of a Simple Recurrent Neural Network (RNN) using TensorFlow Keras for multi-class text classification. The model is trained on the Reuters dataset, which consists of news wire articles categorized into 46 different topics. The objective is to classify news articles into their respective categories.

## Dataset

The project utilizes the `reuters` dataset, a classic dataset for text classification. It comprises short newswire articles from Reuters, each labeled with one of 46 possible topics.

-   **`max_features`**: The number of words to consider as features (set to 10000).
-   **Data Loading**: `tensorflow.keras.datasets.reuters.load_data()` is used to load the dataset, splitting it into training and testing sets (`X_train`, `y_train`, `X_test`, `y_test`).

## Data Preprocessing

1.  **Sequence Padding**: Due to varying lengths of news articles, `tensorflow.keras.preprocessing.sequence.pad_sequences` is applied to standardize the input length for the RNN.
    -   `max_len`: Maximum sequence length (set to 500).
    -   Sequences are padded to `max_len`.
2.  **One-Hot Encoding**: The target labels (`y_train`, `y_test`) are converted to one-hot encoded vectors using `tensorflow.keras.utils.to_categorical` to prepare them for categorical cross-entropy loss.

## Model Architecture

The RNN model is constructed using the `Sequential` API from TensorFlow Keras:

-   **`Embedding` Layer**: Converts integer-encoded words into dense vectors of fixed size.
    -   `input_dim`: Size of the vocabulary (`max_features = 10000`).
    -   `output_dim`: Dimension of the dense embedding (64).
    -   `input_length`: Length of input sequences (`max_len = 500`).
-   **`SimpleRNN` Layer**: A basic recurrent layer processing sequences.
    -   `units`: Dimensionality of the output space (64).
    -   `return_sequences`: `False`, as we are interested in the final output for classification.
-   **`Dense` Layer (1)**: A fully connected layer with ReLU activation.
    -   `units`: 64.
    -   `activation`: `relu`.
-   **`Dense` Layer (2)**: The output layer for classification.
    -   `units`: 46 (corresponding to the 46 Reuters topics).
    -   `activation`: `softmax` for multi-class probability distribution.

## Model Compilation and Training

-   **Optimizer**: `'Adam'` optimizer is used for efficient gradient descent.
-   **Loss Function**: `'categorical_crossentropy'` is chosen, suitable for multi-class classification with one-hot encoded labels.
-   **Metrics**: `'accuracy'` is monitored during training.
-   **Early Stopping**: `EarlyStopping` callback is used to prevent overfitting.
    -   `monitor`: `'val_loss'` (validation loss).
    -   `patience`: 2 (training stops if validation loss does not improve for 2 epochs).
    -   `restore_best_weights`: `True` (restores model weights from the epoch with the best validation loss).

## Usage

To run this notebook:

1.  Ensure TensorFlow is installed (`pip install tensorflow`).
2.  Execute cells sequentially to load data, preprocess, define, compile, and train the model.
3.  The trained model is saved as `model.h5`.
