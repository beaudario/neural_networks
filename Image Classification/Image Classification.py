import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(10, activation='softmax')
])

# Load data from storage to memory
(x, y), (xTest, yTest) = keras.datasets.cifar10.load_data()

# Instantiate the dataset class
trainDataset = tf.data.Dataset.from_tensor_slices((x, y))

for image, label in trainDataset.take(1):
    (image.shape, label.shape)


def augmentation(x, y, HEIGHT=None, WIDTH=None, NUM_CHANNELS=None):
    x = tf.image.resize_image_with_crop_or_pad(
        x, HEIGHT + 8, WIDTH + 8)
    x = tf.image.random_crop(x, [HEIGHT, WIDTH, NUM_CHANNELS])
    x = tf.image.random_flip_left_right(x)
    return x, y


trainDataset = (trainDataset
                .map(augmentation)
                .shuffle(buffer_size=50000)


def normalize(x, y):
    x = tf.image.per_image_standardization(x)
    return x, y


trainDataset = (trainDataset
                .map(augmentation)
                .shuffle(buffer_size=50000)
                .map(normalize))

trainDataset = (trainDataset
                .map(augmentation)
                .shuffle(buffer_size=50000)
                .batch(128, drop_remainder=True))

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.SGD(learning_rate=0.1, momentum=0.9),
    metrics=["accuracy"]
)

model.fit(trainDataset,
          epochs=60,
          validation_data=TestDataset,
          )