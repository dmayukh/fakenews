#hardcode the dataset in the generator
#def data_generator(x, y, batch_size=500):
def data_generator(batch_size=500):    
    counter=0
    num_batches = train_x.shape[0]/batch_size
    if counter == 0:
        shuffle_index = np.arange(np.shape(train_labels)[0])
        np.random.shuffle(shuffle_index)
    while counter < num_batches:
        index_batch = shuffle_index[batch_size*counter:batch_size*(counter+1)]
        counter += 1
        rec = train_x[index_batch, :].todense()
        if len(rec) == batch_size:
            yield rec, train_labels[index_batch]
    counter = 0



def count(stop):
    i = 0
    while i<stop:
        print(i)
        yield i
        i += 1
ds_counter = tf.data.Dataset.from_generator(count, args=[25], output_types=tf.int32, output_shapes = (), )

for count_batch in ds_counter.repeat().batch(8).take(4):
    print(count_batch.numpy())

ds_batcher = tf.data.Dataset.from_generator(batch_generator, args=[train_x, train_labels], output_types=tf.int32, output_shapes = (), )



def gen():
    yield (train_x[:5].todense(), train_labels[:5])
dim = train_x.shape[1]
num_classes = train_labels.shape[1]
ds_batcher = tf.data.Dataset.from_generator(gen, args=[], output_types=(tf.int32, tf.int32) , output_shapes = ((5, dim), (5, num_classes)), )
for count_batch in ds_batcher.repeat().batch(3).take(1):
    print(count_batch)
