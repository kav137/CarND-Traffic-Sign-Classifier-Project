#%%
import functools
from textwrap import wrap
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv
from sklearn.utils import shuffle
#%%
def load_data():
    loaded_data = {}

    training_file = './data/train.p'
    validation_file = './data/valid.p'
    testing_file = './data/test.p'
    labels_file = './signnames.csv'

    with open(training_file, mode='rb') as file:
        train = pickle.load(file)
    with open(validation_file, mode='rb') as file:
        valid = pickle.load(file)
    with open(testing_file, mode='rb') as file:
        test = pickle.load(file)

    loaded_data['x_train'] = train['features']
    loaded_data['y_train'] = train['labels']
    loaded_data['x_valid'] = valid['features']
    loaded_data['y_valid'] = valid['labels']
    loaded_data['x_test'] = test['features']
    loaded_data['y_test'] = test['labels']
    loaded_data['labels'] = np.array(pd.read_csv(labels_file, sep=',').values.tolist())
    loaded_data['labels_number'] = loaded_data['labels'].shape[0]

    return loaded_data

def show_data_info(data):
    assert(
        data['x_train'].shape[1:] == data['x_valid'].shape[1:] and
        data['x_train'].shape[1:] == data['x_test'].shape[1:]
    )
    print('Total number of training images: {}'.format(data['x_train'].shape[0]))
    print('Total number of validation images: {}'.format(data['x_valid'].shape[0]))
    print('Total number of test images: {}'.format(data['x_test'].shape[0]))
    print('Total number of classes: {}'.format(data['labels_number']))
    print('Image size is {} across dataset'.format(data['x_train'].shape[1:]))
#%%
def show_signs_distribution(data):
    plt.subplots(figsize=(12, 15))
    plt.title('Signs distribution across dataset')
    plt.hist(
        [
            data['y_train'],
            data['y_test'],
            data['y_valid']
        ],
        bins=data['labels_number'],
        color=('#1eb53a', '#f77f00', '#b80c09'),
        orientation='horizontal',
        rwidth=0.7,
        label=('Train', 'Test', 'Valid')
    )
    plt.gca().invert_yaxis()
    plt.yticks(np.arange(data['labels_number']), data['labels'][:, 1])
    plt.xlabel('Number of examples')
    plt.legend(loc='lower right')
    plt.show()

def show_signs(data, signs_per_row=5):
    gallery = {'train': [], 'valid': [], 'test':[]}

    signs_info_rows = np.copy(data['labels'])
    np.random.shuffle(signs_info_rows) #pylint: disable=E1101
    signs_info_rows = signs_info_rows[:5]

    for sign_info in signs_info_rows:
        sign_id = int(sign_info[0])
        sign_name = sign_info[1]

        for key, value in gallery.items():
            y_key = 'y_' + key
            x_key = 'x_' + key
            sign_image_indices = np.where(data[y_key] == sign_id)
            sign_image_index = (sign_image_indices[0][0]
                if isinstance(sign_image_indices, tuple)
                else sign_image_indices[0])

            image = data[x_key][sign_image_index]
            gallery_item = [sign_name, image]
            value.append(gallery_item)

    for key, value in gallery.items():
        plt.figure(figsize=(15, 4))
        plt.suptitle('{} dataset'.format(key.capitalize()), size=18)
        for i in range(signs_per_row):
            title, image = value[i]
            plt.subplot(1, signs_per_row, i+1)
            plt.title('\n'.join(wrap(title, 20)))
            plt.imshow(image)

def visualize_data(data):
    show_signs_distribution(data)
    show_signs(data)
#%%
def compose(*fns):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), fns, lambda x: x)

def generate_rotated_image(img, angle_delta=15, return_meta=False):
    height, width = img.shape[:2]
    angle = np.random.randint(-angle_delta, angle_delta + 1)
    scale = 1.25
    rotation_matrix = cv.getRotationMatrix2D((height / 2, width / 2), angle, scale)
    result_image = cv.warpAffine(img, rotation_matrix, (height, width))
    return ([result_image, angle, scale] if return_meta else result_image)

def generate_perspective_transformed_image(img, dim_delta_percentage=.25, return_meta=False):
    height, width = img.shape[:2]
    max_width_delta = dim_delta_percentage * width
    max_height_delta = dim_delta_percentage * height

    initial_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    out_points = np.copy(initial_points)

    for point in out_points:
        x_coord, y_coord = point
        p = np.random.rand(2)
        modify_x, modify_y = np.where(p<0.4, True, False)
        if modify_x:
            x_delta = np.random.randint(0, max_width_delta)
            point[0] = (x_coord + x_delta) if (x_coord > 0) else (x_coord - x_delta)
        if modify_y:
            y_delta = np.random.randint(0, max_height_delta)
            point[1] = (y_coord + y_delta) if (y_coord > 0) else (y_coord - y_delta)

    transform_matrix = cv.getPerspectiveTransform(initial_points, out_points)
    result_image = cv.warpPerspective(img, transform_matrix, (height, width))
    return ([result_image, out_points] if return_meta else result_image)

def generate_blurred_image(img, return_meta=False):
    height, width = img.shape[:2]
    kernel = (7,7)
    blurred_image = cv.GaussianBlur(img, kernel, 0)
    return [blurred_image, kernel] if return_meta else blurred_image

augmentations = [
    generate_blurred_image,
    generate_rotated_image,
    generate_perspective_transformed_image,
    compose(
        generate_blurred_image,
        generate_perspective_transformed_image
    ),
    compose(
        generate_blurred_image,
        generate_rotated_image
    )
]
#%%
def extend_dataset(initial_x, initial_y, labels, labels_total):
    images_per_sign_max = np.max(np.histogram(initial_y, labels_total)[0])
    out_x = np.copy(initial_x)
    out_y = np.copy(initial_y)
    generated_images = []
    generated_labels = []

    for i in range(labels_total):
        sign_id = int(labels[i][0])
        indices_of_sign = np.where(initial_y == sign_id)

        if isinstance(indices_of_sign, tuple):
            indices_to_concat = []
            for indices_arr in indices_of_sign:
                indices_to_concat.append(indices_arr)
            indices_of_sign = np.concatenate(indices_to_concat)

        images_per_sign = indices_of_sign.shape[0]
        divident = 80000
        total_generated_images = int(divident / images_per_sign)
        input_indices = indices_of_sign[:total_generated_images]
        print('Sign {}; IPS : {}; images: {}'.format(sign_id, images_per_sign, total_generated_images))

        for counter in range(total_generated_images):
            index = np.random.choice(input_indices)
            initial_image = initial_x[index]
            augmentation = np.random.choice(augmentations)
            result_image = augmentation(initial_image)
            generated_images.append(result_image)
            generated_labels.append(sign_id)

    out_x = np.concatenate((out_x, generated_images))
    out_y = np.concatenate((out_y, generated_labels))

    return [out_x, out_y]

def grayscale(image):
    gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    return np.array(gray_image).astype(np.float32)

def normalize_gray(gray_image):
    flatten = gray_image.flatten()
    normalized = ((flatten - 128) / 128).reshape((32, 32, 1))
    return normalized

def preprocess_data(raw_data):
    out_data = {}

    extended_x, extended_y = extend_dataset(
        raw_data['x_train'], raw_data['y_train'], raw_data['labels'], raw_data['labels_number']
    )

    convert_images_fn = compose(normalize_gray, grayscale)
    out_data['x_train'] = np.array(list(map(convert_images_fn, extended_x)))
    out_data['x_test'] = np.array(list(map(convert_images_fn, raw_data['x_test'])))
    out_data['x_valid'] =  np.array(list(map(convert_images_fn, raw_data['x_valid'])))

    out_data['y_train'] = extended_y
    out_data['y_test'] = raw_data['y_test']
    out_data['y_valid'] = raw_data['y_valid']

    out_data['labels'] = raw_data['labels']
    out_data['labels_number'] = raw_data['labels_number']

    return out_data

#%%
input_data = load_data()
show_data_info(input_data)
visualize_data(input_data)
#%%
preprocessed_data = preprocess_data(input_data)
#%%
plt.hist(input_data['y_train'], bins=43)
plt.show()
plt.hist(preprocessed_data['y_train'], bins=43)
plt.show()
#%%
print(preprocessed_data['x_train'].shape)
#%%
test_image = input_data['x_train'][100]
rotated_image, rotation_angle, scale = generate_rotated_image(test_image, return_meta=True)
skewed_image, out_matrix = generate_perspective_transformed_image(test_image, return_meta=True)
blurred_image = generate_blurred_image(test_image)
plt.figure(figsize=(5,5))
plt.title('original')
plt.imshow(test_image)
plt.figure(figsize=(5,5))
plt.title('rotated (angle: {}, scale: {})'.format(rotation_angle, scale))
plt.imshow(rotated_image)
plt.figure(figsize=(5,5))
plt.title('\n'.join(wrap('skewed, (new coords for corners: {})'.format(out_matrix), 50)))
plt.imshow(skewed_image)
plt.figure(figsize=(5,5))
plt.title('blurred')
plt.imshow(blurred_image)
#%%
EPOCHS = 25
BATCH_SIZE = 128
mu = 0
sigma = 0.1
features_total = preprocessed_data['labels_number']

def model(x):
    padding_conv = 'VALID'
    padding_pool = 'SAME'

    strides = {
        'conv_1': [1, 1, 1, 1],
        'conv_2': [1, 1, 1, 1],
        'pool_1': [1, 2, 2, 1],
        'pool_2': [1, 2, 2, 1]
    }

    weights = {
        'conv_1': tf.Variable(tf.truncated_normal([5, 5, 1, 16], mean=mu, stddev=sigma)),
        'conv_2': tf.Variable(tf.truncated_normal([5, 5, 16, 32], mean=mu, stddev=sigma)),
        'ful_1': tf.Variable(tf.truncated_normal([800, 300], mean=mu, stddev=sigma)),
        'ful_2': tf.Variable(tf.truncated_normal([300, 70], mean=mu, stddev=sigma)),
        'ful_3': tf.Variable(tf.truncated_normal([70, features_total], mean=mu, stddev=sigma)),
    }

    biases = {
        'conv_bias_1': tf.Variable(tf.zeros(16)),
        'conv_bias_2': tf.Variable(tf.zeros(32)),
        'ful_bias_1': tf.Variable(tf.zeros(300)),
        'ful_bias_2': tf.Variable(tf.zeros(70)),
        'ful_bias_3': tf.Variable(tf.zeros(features_total)),
    }

    kernels = {
        'pool_1': [1, 2, 2, 1],
        'pool_2': [1, 2, 2, 1]
    }

    conv_1 = tf.nn.conv2d(x, weights['conv_1'], strides['conv_1'], padding_conv)
    conv_1 = tf.nn.bias_add(conv_1, biases['conv_bias_1'])
    activation_1 = tf.nn.relu(conv_1)
    pool_1 = tf.nn.max_pool(activation_1, kernels['pool_1'], strides['pool_1'], padding_pool)
    layer_1 = pool_1

    conv_2 = tf.nn.conv2d(layer_1, weights['conv_2'], strides['conv_2'], padding_conv)
    conv_2 = tf.nn.bias_add(conv_2, biases['conv_bias_2'])
    activation_2 = tf.nn.relu(conv_2)
    pool_2 = tf.nn.max_pool(activation_2, kernels['pool_2'], strides['pool_2'], padding_pool)
    layer_2 = pool_2
    layer_2 = tf.nn.dropout(layer_2, 0.75)

    flattened = tf.contrib.layers.flatten(layer_2)

    layer_3 = tf.matmul(flattened, weights['ful_1'])
    layer_3 = tf.nn.bias_add(layer_3, biases['ful_bias_1'])
    layer_3 = tf.nn.relu(layer_3)

    layer_4 = tf.matmul(layer_3, weights['ful_2'])
    layer_4 = tf.nn.bias_add(layer_4, biases['ful_bias_2'])
    layer_4 = tf.nn.sigmoid(layer_4)

    layer_5 = tf.matmul(layer_4, weights['ful_3'])
    layer_5 = tf.nn.bias_add(layer_5, biases['ful_bias_3'])

    logits = layer_5

    return logits
#%%
### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected,
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, features_total)

rate = 0.001
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(0.001, global_step, 800, 0.8, staircase=True)

logits = model(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_operation = optimizer.minimize(loss_operation, global_step=global_step)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(x_data, y_data):
    num_examples = len(x_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = x_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(preprocessed_data['x_train'])

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(preprocessed_data['x_train'], preprocessed_data['y_train'])
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(preprocessed_data['x_valid'], preprocessed_data['y_valid'])

        print("EPOCH {} ...".format(i+1))
        print('Learning rate = {}'.format(sess.run(learning_rate)))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './lenet')
    print("Model saved")

    final_accuracy = evaluate(preprocessed_data['x_test'], preprocessed_data['y_test'])
    print("Result Accuracy = {:.3f}".format(final_accuracy))
