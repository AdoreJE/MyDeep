import os
from glob import glob

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

nb_classes = 2
X = tf.placeholder(tf.float32, [None, 2764800])
Y = tf.placeholder(tf.float32, [None, nb_classes])

W = tf.Variable(tf.random_normal([2764800, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

h = tf.nn.softmax(tf.matmul(X,W)+b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(h), axis=1))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Test model
is_correct = tf.equal(tf.argmax(h, 1), tf.argmax(Y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# parameters
num_epochs = 15
batch_size = 100
num_iterations = int(mnist.train.num_examples / batch_size)

with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    # Training cycle
    for epoch in range(num_epochs):
        avg_cost = 0

        for i in range(num_iterations):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, cost_val = sess.run([train, cost], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += cost_val / num_iterations

        print("Epoch: {:04d}, Cost: {:.9f}".format(epoch + 1, avg_cost))

    print("Learning finished")

    # Test the model using test sets
    print(
        "Accuracy: ",
        accuracy.eval(
            session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}
        ),
    )

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r : r + 1], 1)))
    print(
        "Prediction: ",
        sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r : r + 1]}),
    )

    plt.imshow(
        mnist.test.images[r : r + 1].reshape(28, 28),
        cmap="Greys",
        interpolation="nearest",
    )
    plt.show()


data_list = glob('data/train/*/*.jpg')
path = data_list[0]

def get_label_from_path(path):
	if path.split('/')[-2] == 'accident':
		return 1
	elif path.split('/')[-2] == 'normal':
		return 0

def read_image(path):
	image = np.array(Image.open(path))
	return image.reshape(image.shape[0], image.shape[1], 3)

class_name = get_label_from_path(path)

label_name_list = []

for path in data_list:
	label_name_list.append(get_label_from_path(path))

unique_label_names = np.unique(label_name_list)

def onehot_encode_label(path):
	onehot_label = unique_label_names == get_label_from_path(path)
	onehot_label = onehot_label.astype(np.uint8)
	return onehot_label

print(onehot_encode_label(path))

# Hyper Parameter 
batch_size = 64
data_height = 720
data_width = 1280
channel_n = 3
num_classes = 2

# 방법.1 - Empty Array를 만들고 채워가는 방법
batch_image = np.zeros((batch_size, data_height, data_width, channel_n))
batch_label = np.zeros((batch_size, num_classes))

# 간단한 batch data 만들기
for n, path in enumerate(data_list[:batch_size]):
    image = read_image(path)
    onehot_label = onehot_encode_label(path)
    batch_image[n, :, :, :] = image
    batch_label[n, :] = onehot_label
print(batch_image.shape, batch_label.shape)


batch_per_epoch = batch_size // len(data_list)
# for batch_n in range(batch_per_epoch):
#     batch_data = data_list[batch_n * batch_size : (batch_n + 1) * batch_size]

def _read_py_function(path, label):
    image = read_image(path)
    label = np.array(label, dtype=np.uint8)
    return image.astype(np.int32), label
def _resize_function(image_decoded, label):
    image_decoded.set_shape([None, None, None])
    image_resized = tf.image.resize_images(image_decoded, [28, 28])
    return image_resized, label
# label을 array 통채로 넣는게 아니고, list 화 시켜서 하나씩 넣기 위해 list로 바꿔주었다. 
label_list = [onehot_encode_label(path).tolist() for path in data_list]
dataset = tf.data.Dataset.from_tensor_slices((data_list, label_list))
dataset = dataset.map(
    lambda data_list, label_list: tuple(tf.py_func(_read_py_function, [data_list, label_list], [tf.int32, tf.uint8])))
dataset = dataset.map(_resize_function)
dataset = dataset.repeat()
dataset = dataset.shuffle(buffer_size=(int(len(data_list) * 0.4) + 3 * batch_size))
dataset = dataset.batch(batch_size)
iterator = dataset.make_initializable_iterator()
image_stacked, label_stacked = iterator.get_next()
next_element = iterator.get_next()
with tf.Session() as sess:
    sess.run(iterator.initializer)
    image, label = sess.run([image_stacked, label_stacked])

print(image, label)

