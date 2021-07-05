import numpy as np
import tensorflow as tf

x_shape = (14000, 1001)
y_shape = (14000,)  # A single item (not array).
classes = 3

x = np.random.random_sample(x_shape).astype(np.float32)
#create a sparse matrix
x=sparse.csr_matrix(x)
y = np.random.randint(0, classes, size=y_shape, dtype=np.int32)
labels_3 = np.zeros(shape=(len(y),3))
for idx, val in enumerate(y):
    labels_3[idx][val]=1
                
# This is tf.data.experimental.AUTOTUNE in older tensorflow.
AUTOTUNE = tf.data.AUTOTUNE

def generator_fn(n_samples):
    """Return a function that takes no arguments and returns a generator."""
    def generator():
        num_batches = x.shape[0]/n_samples
        counter = 0
        if counter == 0:
            idx = np.arange(x.shape[0])
            np.random.shuffle(idx)
        
        while counter < num_batches:
            index_batch = idx[n_samples*counter:n_samples*(counter+1)]
            #yield x[index_batch].todense(), labels_3[index_batch]
            counter += 1
            rec = x[index_batch, :].todense()
            if len(rec) == n_samples:
                yield rec, labels_3[index_batch]
        counter = 0

    return generator

# def augment(x, y):
#     return x * tf.random.normal(shape=x_shape), y

samples = 100
batch_size = 1
epochs = 10

# Create dataset.
gen = generator_fn(n_samples=samples)
dataset = tf.data.Dataset.from_generator(
    generator=gen, 
    output_types=(np.float32, np.int32), 
    output_shapes=((samples, 1001), (samples, 3))
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
#model = tf.keras.applications.VGG16(weights=None, input_shape=x_shape, classes=classes)
inp = keras.Input(shape=(None, 1001), sparse=False)
x1 = Dense(100, activation='relu')(inp)
x2 = Dropout(0.4)(x1)
x3 = keras.layers.Dense(3, activation='softmax')(x2)
model = keras.Model(inp, x3)
model.compile(loss='categorical_crossentropy',
          optimizer=tf.keras.optimizers.Adam(lr=0.001), 
          metrics=['accuracy'])
model.summary()

#model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# Train. Do not specify batch size because the dataset takes care of that.
model.fit(dataset, epochs=epochs)

