import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Concatenate, SimpleRNN
from tensorflow.keras.models import Model

### FFNet

def get_ffnet(n_features):
    
    """
        Creates and returns BetaNet, a simple feedforward network created
        to being testing the BCINet paradigm. Created using the Keras functional API.
        Args: len_azimuth --- determines how many features we'll use.
        Returns: ffnet model
    """
    
    ### AV Input --> V1 & A1 --> Combined in PT --> Into Dense FT Layer --> Output
    
    # Input Layers
    vis_input = Input(shape = (n_features,), name = "VIS") # Visual Info
    aud_input = Input(shape = (n_features,), name = "AUD") # Auditory Info

    # V1 - Unisensory Vision
    V1 = Dense(8, activation = "relu", name = "V1")(vis_input)
    V1 = Model(inputs = vis_input, outputs = V1)
    
    # A1 - Unisensory Audition
    A1 = Dense(8, activation = "relu", name = "A1")(aud_input)
    A1 = Model(inputs = aud_input, outputs = A1)

    # Parietal-Temporal - Forced Fusion?
    
    PT = Concatenate()([V1.output, A1.output])
    PT = Dense(8, activation = "relu", name = "PT")(PT)
    
    # Frontal - Causal Inference?
    FT = Dense(8, activation = "relu", name = "FT")(PT)
    
    # Output?
    OUT_VIS = Dense(5, activation = "sigmoid", name = "VOUT")(FT)
    OUT_AUD = Dense(5, activation = "sigmoid", name = "AOUT")(FT)
    
    # BetaNet
    ffnet = Model(inputs = [vis_input, aud_input], outputs = [OUT_VIS, OUT_AUD])
    ffnet.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return ffnet
