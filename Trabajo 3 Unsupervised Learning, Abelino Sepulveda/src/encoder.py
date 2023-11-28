import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def autoencoder_dimension_reduction_ampliation(data_scaled: np.ndarray, reduce_dimensions: bool = True) -> np.ndarray:
    """
    Reducción de dimensionalidad utilizando un autoencoder.

    Args:
        data_scaled (array): Matriz de datos escalados de forma (n_samples, n_features).
        reduce_dimensions (bool): Indica si se debe reducir la dimensionalidad (True) o aumentarla (False).

    Returns:
        array: Características codificadas resultantes de la reducción de dimensionalidad.
    """

    X_train, X_test = train_test_split(data_scaled, test_size=0.2, random_state=42)

    input_dim = X_train.shape[1]
    output_dim = input_dim // 2 if reduce_dimensions else input_dim * 2

    encoder = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_dim,)),
        tf.keras.layers.Dense(output_dim, activation='relu'),
    ])

    decoder = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(output_dim,)),
        tf.keras.layers.Dense(input_dim, activation='linear'),
    ])

    autoencoder = tf.keras.Sequential([encoder, decoder])

    autoencoder.compile(optimizer='adam', loss='mse') 
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=16, shuffle=True, validation_data=(X_test, X_test))
    encoded_features = encoder.predict(X_test)

    return encoded_features
