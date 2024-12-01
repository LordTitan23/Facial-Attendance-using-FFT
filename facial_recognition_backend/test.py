import tensorflow as tf
import numpy as np

# Create a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(226, 226, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1)
])

# Create a random image with shape (1, 226, 226, 3)
image = np.random.rand(1, 226, 226, 3).astype(np.float32)

# Perform a forward pass
output = model(image)
print(output)
