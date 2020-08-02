from keras_preprocessing.image import ImageDataGenerator
import tensorflow.keras as k
import tensorflow.keras.backend as K
import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import metrics
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model

K.set_image_data_format('channels_last')
PATH = 'D:\\How To\\Data Science\\Projects\\data\\Face Mask Classification'
IMAGE_SIZE = 96
EPOCHS = 50


def create_model(input_shape):
    """
    Implementation of the HappyModel.

    Arguments:
    input_shape -- shape of the images of the dataset
        (height, width, channels) as a tuple.
        Note that this does not include the 'batch' as a dimension.
        If you have a batch like 'X_train',
        then you can provide the input_shape using
        X_train.shape[1:]


    Returns:
    model -- a Model() instance in Keras
    """
    X_input = Input(input_shape)

    # X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(16, (4, 4), strides=(1, 1), name='conv0')(X_input)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool1')(X)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(16, (3, 3), strides=(1, 1), name='conv1')(X)
    X = BatchNormalization(axis=3, name='bn1')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool2')(X)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (3, 3), strides=(1, 1), name='conv2')(X)
    X = BatchNormalization(axis=3, name='bn2')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool3')(X)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (3, 3), strides=(1, 1), name='conv3')(X)
    X = BatchNormalization(axis=3, name='bn3')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool4')(X)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (3, 3), strides=(1, 1), name='conv4')(X)
    X = BatchNormalization(axis=3, name='bn4')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool5')(X)

    # FLATTEN X + FC layer
    X = Flatten()(X)
    X = Dense(2, activation='relu', name='fc')(X)

    model = Model(inputs=X_input, outputs=X, name='cnn_model')

    return model


def CNN_test_with_data_split():
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    dev_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(
        directory=os.path.join(PATH, 'train'),
        target_size=(96, 96),
        color_mode="rgb",
        classes=None,
        class_mode="categorical",
        batch_size=32,
        shuffle=True,
        seed=None,
        save_to_dir=None,
        save_prefix="",
        save_format="png",
        follow_links=False,
        interpolation="nearest")

    validation_generator = dev_datagen.flow_from_directory(
        directory=os.path.join(PATH, 'dev'),
        target_size=(96, 96),
        color_mode="rgb",
        classes=None,
        class_mode="categorical",
        batch_size=32,
        shuffle=True,
        seed=None,
        save_to_dir=None,
        save_prefix="",
        save_format="png",
        follow_links=False,
        interpolation="nearest")

    test_generator = test_datagen.flow_from_directory(
        directory=os.path.join(PATH, 'test'),
        target_size=(96, 96),
        color_mode="rgb",
        classes=None,
        class_mode="categorical",
        batch_size=32,
        shuffle=True,
        seed=None,
        save_to_dir=None,
        save_prefix="",
        save_format="png",
        follow_links=False,
        interpolation="nearest")

    model = create_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    model.summary()

    optimizer = k.optimizers.Adam(learning_rate=0.01)
    model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=[
        metrics.MeanSquaredError(name='my_mse')
    ])

    checkpoint = ModelCheckpoint("model-{val_iou:.2f}.h5", monitor="val_iou", verbose=1, save_best_only=True,
                                 save_weights_only=True, mode="max")

    stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-4, verbose=1, mode="min")

    model.fit(x=train_generator,
              validation_data=validation_generator,
              epochs=EPOCHS,
              callbacks=[reduce_lr, stop],
              workers=1,
              use_multiprocessing=False,
              shuffle=True,
              verbose=1)

    model.save('cnn_model.hdf5')

    model.evaluate(x=test_generator,
                   workers=1,
                   use_multiprocessing=False,
                   verbose=1)


if __name__ == "__main__":
    CNN_test_with_data_split()
