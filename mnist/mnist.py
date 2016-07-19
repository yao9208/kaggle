import tensorflow as tf
import pandas as pd
import numpy as np

BATCH_SIZE = 100
data = pd.read_csv('./train.csv')
images = data.iloc[:,1:].values
images = images.astype(np.float)

images = np.multiply(images, 1.0/255.0)

train_labels = data[[0]].values.ravel()


epochs_completed = 0
index_in_epoch = 0
num_examples = images.shape[0]

# serve data by batches
def next_batch(batch_size):

    global images
    global labels
    global index_in_epoch
    global epochs_completed

    start = index_in_epoch
    index_in_epoch += batch_size

    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        images = images[perm]
        labels = labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return images[start:end], labels[start:end]

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
sess = tf.InteractiveSession()

labels = dense_to_one_hot(train_labels, 10)
labels = labels.astype(np.uint8)

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

tf.initialize_all_variables().run()
for i in range(1000):
    batch_xs, batch_ys = next_batch(BATCH_SIZE)
    train_step.run(feed_dict={x: batch_xs, y_: batch_ys})



#test
test_images = pd.read_csv('./test.csv').values
test_images = test_images.astype(np.float)

# convert from [0:255] => [0.0:1.0]
test_images = np.multiply(test_images, 1.0 / 255.0)
# using batches is more resource efficient
predicted_lables = np.zeros(test_images.shape[0])
predict = tf.argmax(y,1)
for i in range(0,test_images.shape[0]//BATCH_SIZE):
    predicted_lables[i*BATCH_SIZE : (i+1)*BATCH_SIZE] = predict.eval(feed_dict={x: test_images[i*BATCH_SIZE : (i+1)*BATCH_SIZE]})

np.savetxt('submission_softmax.csv',
           np.c_[range(1,len(test_images)+1),predicted_lables],
           delimiter=',',
           header = 'ImageId,Label',
           comments = '',
           fmt='%d')
