import numpy as np
import tensorflow as tf

x_shape = (32, 32, 3)
y_shape = ()  # A single item (not array).
classes = 10

# This is tf.data.experimental.AUTOTUNE in older tensorflow.
AUTOTUNE = tf.data.AUTOTUNE

def generator_fn(n_samples):
    """Return a function that takes no arguments and returns a generator."""
    def generator():
        for i in range(n_samples):
            # Synthesize an image and a class label.
            x = np.random.random_sample(x_shape).astype(np.float32)
            y = np.random.randint(0, classes, size=y_shape, dtype=np.int32)
            yield x, y
    return generator

# def augment(x, y):
#     return x * tf.random.normal(shape=x_shape), y

samples = 10
batch_size = 5
epochs = 2

# Create dataset.
gen = generator_fn(n_samples=samples)
dataset = tf.data.Dataset.from_generator(
    generator=gen, 
    output_types=(np.float32, np.int32), 
    output_shapes=(x_shape, y_shape)
)
# Parallelize the augmentation.
# dataset = dataset.map(
#     augment, 
#     num_parallel_calls=AUTOTUNE,
#     # Order does not matter.
#     deterministic=False
# )
dataset = dataset.batch(batch_size, drop_remainder=True)
# Prefetch some batches.
#dataset = dataset.prefetch(AUTOTUNE)

# Prepare model.
model = tf.keras.applications.VGG16(weights=None, input_shape=x_shape, classes=classes)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# Train. Do not specify batch size because the dataset takes care of that.
model.fit(dataset, epochs=epochs)


