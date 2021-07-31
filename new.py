
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

data_x = [
 'good',  'well done', 'nice', 'Excellent',
 'Bad', 'OOps I hate it deadly', 'embrassing', 
'A piece of shit']

label_x = np.array([1,1,1,1, 0,0,0,0])

# one hot encoding 

one_hot_x = [tf.keras.preprocessing.text.one_hot(d, 50) for d in data_x]

# padding 

padded_x = tf.keras.preprocessing.sequence.pad_sequences(one_hot_x, maxlen=4, padding = 'post')

# Architecting our Model 

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(50, 8, input_length=4),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
 ])

# specifying training params 
 
model.compile(optimizer='adam', loss='binary_crossentropy', 
metrics=['accuracy'])

history = model.fit(padded_x, label_x, epochs=1000, 
batch_size=2, verbose=0)

# plotting training graph

plt.plot(history.history['loss'])