import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load and preprocess the dataset (replace this with your dataset loading and preprocessing)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
y_train = to_categorical(y_train, 26)  # Change the number of classes to 26
y_test = to_categorical(y_test, 26)   # Change the number of classes to 26

# Define the Residual Block
def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x

    # First convolution
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Second convolution
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Shortcut connection
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)

    return x

# Define the ResNet model
def build_resnet(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)

    # Initial convolution
    x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

    # Residual blocks
    num_blocks_list = [3, 4, 6, 3]
    for i, num_blocks in enumerate(num_blocks_list):
        stride = 2 if i > 0 else 1
        x = residual_block(x, 64 * (2 ** i), stride=stride)
        for _ in range(num_blocks - 1):
            x = residual_block(x, 64 * (2 ** i))

    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Fully connected layer
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, x, name='resnet')
    return model

# Set input shape and number of classes
input_shape = (28, 28, 1)  # Adjust according to your input image size
num_classes = 26  # Change the number of classes to 26 for English letters

# Build the model
model = build_resnet(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

# Train the model
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test))
