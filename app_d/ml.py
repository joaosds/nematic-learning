#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 10:04:52 2022

@author: obernauer
"""


import os
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
from stm import STM


import matplotlib
from matplotlib import rc
import matplotlib.ticker
import scipy

rc("font", family="serif", serif="cm10")
matplotlib.rcParams["text.usetex"] = True
# matplotlib.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]

matplotlib.rcParams["axes.linewidth"] = 2.4  # width of frames


class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=False, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(
            self, useOffset=offset, useMathText=mathText
        )

    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom

    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r"$\:\mathdefault{%s}$" % self.format


def stat(y_test, pred):
    y_test, pred = np.array(y_test), np.array(pred)
    mape = np.round(np.mean(np.abs((y_test - pred) / y_test)), 3)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_test, pred)
    rsq = np.round(r_value**2, 3)
    return mape, rsq


def process_images(image):
    """
    Add additional channel axis to image. Convert image from float64 to float32.
    Standardize image values. Add gaussian noise.
    https://towardsdatascience.com/implementing-alexnet-cnn-architecture-using-
    tensorflow-2-0-and-keras-2113e090ad98?gi=164148c31297
    """
    # image = image / 100.
    image = image[..., tf.newaxis]  # [height, width, channels]
    image = tf.image.convert_image_dtype(image, tf.float32)
    # Normalize images to have a mean of 0 and standard deviation of 1
    # image = tf.image.per_image_standardization(image)
    # Resize images from 67x67 to 227x227
    # image = tf.image.resize(image, (227,227))#, 'nearest')
    batch, row, col, ch = image.shape
    mean = 0
    sigma = 0.05
    gauss = np.abs(np.random.normal(mean, sigma, (batch, row, col, ch)))
    gauss = gauss.reshape(batch, row, col, ch)
    image = image + gauss
    return image


def get_gaussian_noise_output(model, epoch) -> None:
    """
    Plotting the output of a gaussian noise layer.

    Parameters
    ----------
    model :
        Model with gaussian noise.
    epoch : int
        count for number of trained epochs.
    """
    with np.load("training.npz") as data:
        image = data["DataX"][0]
        image2 = data["DataX"][1]
        image3 = data["DataX"][2]
        image4 = data["DataX"][3]
        label = data["DataY"][0]

    partial_model = keras.Model(model.inputs, model.get_layer("gauss_x").output)

    output_train = partial_model(
        (image, image2, image3, image4), training=True
    )  # runs the model in training mode

    cmap = plt.cm.coolwarm.copy()
    if epoch == 0:
        plt.imshow(image, cmap=cmap)
        plt.title(
            "Original with " r"$\alpha = $" + str(np.round(label, 3)), fontsize=10
        )
        plt.colorbar()
        plt.axis("off")
        plt.show()
    plt.imshow(output_train, cmap=cmap)
    plt.title(
        "Added Gaussian noise with "
        r"$\sigma = $" + str(np.round(model.get_layer("gauss_x").stddev, 2)),
        fontsize=10,
    )
    plt.colorbar()
    plt.axis("off")
    plt.show()


def plot_history() -> None:
    """
    The fit() method returns a History object containing the training
    parameters (history.params), the list of epochs it went through
    (history.epoch), and most importantly a dictionary (history.history)
    containing the loss and extra metrics it measured at the end of each
    epoch on the training set and on the validation set.

    Here I plot the learning curves.
    """
    with open("history.pkl", "rb") as file:
        history = pickle.load(file)
    loss, acc, val_loss, val_acc = list(history.values())
    start = 50  # at which epoch the plot should start
    plt.plot(range(start, len(loss)), loss[start:], "r", label="training loss")
    plt.plot(
        range(start, len(val_loss)), val_loss[start:], "b", label="validation loss"
    )
    plt.legend()
    plt.grid(True)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
    plt.plot(range(1, len(acc) + 1), acc, "r", label="training accuracy")
    plt.plot(range(1, len(val_acc) + 1), val_acc, "b", label="validation accuracy")
    plt.legend()
    plt.grid(True)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()


class GaussianNoiseCallback(tf.keras.callbacks.Callback):
    """
    Class for adjusting the standard dev of the gaussian layer after every epoch.
    """

    def on_epoch_begin(self, epoch, logs=None):
        """
        Adjusting the std dev and plotting an example via get_gaussian_noise_output()

        Parameters
        ----------
        Are needed due to class inheritance.
        """
        # stddev = random.uniform(0, 1)
        self.model.get_layer("gauss_x").stddev = random.uniform(0, 1)
        self.model.get_layer("gauss_y").stddev = random.uniform(0, 1)
        self.model.get_layer("gauss_a").stddev = random.uniform(0, 1)
        self.model.get_layer("gauss_b").stddev = random.uniform(0, 1)
        # get_gaussian_noise_output(self.model, epoch)


class ML(STM):
    """
    Subclass for extracting nematic order parameters from STM image.

    Attributes
    ----------
    model_name : str
        Name under which model is stored or loaded.
    batch_size : int
        A tf.int64 scalar tf.Tensor, representing the number of consecutive
        elements of this dataset to combine in a single batch.
    epochs : int
        Number of epochs to train the model. An epoch is an iteration over
        the entire x and y data provided.
    model :
        Either a loaded model from h5 file or a newly generated one.
    train_ds :
        training data
    val_ds :
        validation data
    test_ds :
        test data
    history :
        here learning curves are stored.

    Methods
    -------
    process_training_data :
        Reading in and processing training and validation data from npz files
        on hard disk.
    process_test_data :
        Reading in and processing test data from npz files on hard disk.
    plot_training_data :
        Plotting examples of the training data.
    create_model :
        Define a sequential model, the optimizer and loss function.
    train_model :
        Train the defined model from create_model using callbacks.
    evaluate_model :
        Evaluate model on the test set to estimate the generalization error.
    analyize_model :
        Analyze accuracy of model.
    image_error :
        Calculate error between Image(class_true) and Image(class_predicted)
    """

    def __init__(self, model: str, batch_size: int, epochs: int) -> None:
        """
        Constructs all the necessary attributes for machine learning class.

        Parameters
        ----------
        model : str
            Check for stored model with name 'model' in h5 format.
        batch_size : int
            A tf.int64 scalar tf.Tensor, representing the number of consecutive
            elements of this dataset to combine in a single batch.
        epochs : int
            Number of epochs to train the model. An epoch is an iteration over
            the entire x and y data provided.
        """
        self.model_name = model
        if os.path.exists(str(self.model_name)):
            self.model = keras.models.load_model(self.model_name)
            print("\nLoaded model: ", self.model_name)
            print(self.model.summary())
        else:
            self.model = False

        self.batch_size = batch_size
        self.epochs = epochs

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        self.history = None

        super().__init__()

    def process_training_data(self):
        """
        Reading in and processing training and validation data from
        npz files on hard disk.
        """
        with np.load("trafo_train.npz") as data:
            trafo_train = data["DataX"].astype(np.float32)[..., tf.newaxis]

        with np.load("training.npz") as data:
            # train_examples_1 = data['DataX'][0:4*len(trafo_train):4]
            # train_examples_2 = data['DataX'][1:4*len(trafo_train)+1:4]
            # train_examples_3 = data['DataX'][2:4*len(trafo_train)+2:4]
            # train_examples_4 = data['DataX'][3:4*len(trafo_train)+3:4]
            train_labels = data["DataY"].astype(np.float32)[
                0 : 4 * len(trafo_train) : 4
            ]

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (
                (
                    # process_images(train_examples_1),
                    # process_images(train_examples_2),
                    # process_images(train_examples_3),
                    # process_images(train_examples_4),
                    trafo_train
                ),
                train_labels,
            )
        )
        train_ds_size = tf.data.experimental.cardinality(train_dataset).numpy()
        print("Training data size of one energy channel:", train_ds_size)
        self.train_ds = (
            train_dataset.shuffle(buffer_size=train_ds_size)
            .batch(batch_size=self.batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )

        with np.load("trafo_val.npz") as data:
            trafo_val = data["DataX"].astype(np.float32)[..., tf.newaxis]

        with np.load("validation.npz") as data:
            # val_examples_1 = data['DataX'][0:4*len(trafo_val):4]
            # val_examples_2 = data['DataX'][1:4*len(trafo_val)+1:4]
            # val_examples_3 = data['DataX'][2:4*len(trafo_val)+2:4]
            # val_examples_4 = data['DataX'][3:4*len(trafo_val)+3:4]
            val_labels = data["DataY"].astype(np.float32)[0 : 4 * len(trafo_val) : 4]

        val_dataset = tf.data.Dataset.from_tensor_slices(
            (
                (
                    # process_images(val_examples_1),
                    # process_images(val_examples_2),
                    # process_images(val_examples_3),
                    # process_images(val_examples_4),
                    trafo_val
                ),
                val_labels,
            )
        )
        val_ds_size = tf.data.experimental.cardinality(val_dataset).numpy()
        print("Validation data size of one energy channel:", val_ds_size)
        self.val_ds = (
            val_dataset.shuffle(buffer_size=val_ds_size)
            .batch(batch_size=self.batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )

    def process_test_data(self) -> None:
        """
        Reading in and processing training and validation data from
        npz files on hard disk.
        """
        # with np.load("trafo_test.npz") as data:
        with np.load("appd_trafo_test.npz") as data:
            trafo_test = data["DataX"].astype(np.float32)[..., tf.newaxis]

        with np.load("appd_test.npz") as data:
            test_examples_1 = data["DataX"][0 : 4 * len(trafo_test) : 4]
            test_examples_2 = data["DataX"][1 : 4 * len(trafo_test) + 1 : 4]
            test_examples_3 = data["DataX"][2 : 4 * len(trafo_test) + 2 : 4]
            test_examples_4 = data["DataX"][3 : 4 * len(trafo_test) + 3 : 4]
            test_labels = data["DataY"].astype(np.float32)[0 : 4 * len(trafo_test) : 4]

        test_dataset = tf.data.Dataset.from_tensor_slices(
            (
                (
                    process_images(test_examples_1),
                    process_images(test_examples_2),
                    process_images(test_examples_3),
                    process_images(test_examples_4),
                    trafo_test,
                ),
                test_labels,
            )
        )
        test_ds_size = tf.data.experimental.cardinality(test_dataset).numpy()
        print("Test data size of one energy channel:", test_ds_size)
        self.test_ds = test_dataset.batch(batch_size=1, drop_remainder=True)

    def plot_training_data(self) -> None:
        """
        Plotting examples of the training data.
        """
        cmap = plt.cm.coolwarm.copy()
        for images, labels in self.train_ds.take(1):
            for i in range(1):
                plt.title(np.round(labels[i].numpy(), 4), fontsize=12)
                plt.imshow(images[0][i].numpy(), cmap=cmap)
                plt.colorbar()
                plt.axis("off")
                plt.show()
                plt.imshow(images[4][i].numpy(), cmap=cmap)
                plt.colorbar()
                plt.axis("off")
                plt.show()

    def create_model(self) -> None:
        """
        A sequential model: This is the simplest kind of Keras model,
        for neural networks that are just composed of a single stack of layers,
        connected sequentially
        ----------------------------------------------------------------------
        Flatten layer: converts each input image into a 1D array. If it receives
        input data X, it computes X.reshape(-1, 1). This layer does not have
        any parameters, it is just there to do some simple preprocessing.

        If it is the first layer in the model, one should specify the
        input_shape: this does not include the batch size, only the shape of
        the instances. Alternative: add a keras.layers.InputLayer as the
        first layer.
        ----------------------------------------------------------------------
        Dense layer: here it will use the ReLU activation function.
        Each Dense layer manages its own weight matrix, containing all the
        connection weights between the neurons and their inputs. It also
        manages a vector of bias terms (one per neuron). When it receives
        some input data, it computes h_W,b (X) = act(XW + b)
        Dense layers often have a lot of parameters (e.g. 9216 * 128 conncection
                                                     weights + 126 bias terms)
        -risk of overfitting
        ----------------------------------------------------------------------
        Convolutional layer:  neurons in the first convolutional layer are not
        connected to every single pixel in the input image (like they in Dense
        layers), but only to pixels in their receptive fields. In turn, each
        neuron in the second convolutional layer is connected only to neurons
        located within a small rectangle in the first layer.
        ----------------------------------------------------------------------
        Pooling layer: Just like in convolutional layers, each neuron in a
        pooling layer is connected to the outputs of a limited number of neurons
        in the previous layer, located within a small rectangular receptive
        field. One must define its size, the stride, and the padding type,
        just like before. However, a pooling neuron has no weights; all it does
        is aggregate the inputs using an aggregation function (e.g. max).
        ----------------------------------------------------------------------
        After a model is created, one calls its compile() method to specify
        the loss function and the optimizer to use. One can also specify a list
        of extra metrics to compute during training and evaluation. Since this
        is a classifier, it’s useful to measure its "accuracy" during training
        and evaluation
        """
        # self.model = keras.models.Sequential([
        #     keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(3,3),
        #                         activation='relu', input_shape=(65,65,1)),
        #     #keras.layers.BatchNormalization(),
        #     keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        #     keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1),
        #                         activation='relu', padding="same"),
        #     #keras.layers.BatchNormalization(),
        #     keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        #     keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1),
        #                         activation='relu', padding="same"),
        #     #keras.layers.BatchNormalization(),
        #     keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1),
        #                         activation='relu', padding="same"),
        #     #keras.layers.BatchNormalization(),
        #     keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1),
        #                         activation='relu', padding="same"),
        #     #keras.layers.BatchNormalization(),
        #     keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        #     keras.layers.Flatten(),
        #     keras.layers.Dense(4096, activation='relu'),
        #     #keras.layers.Dropout(0.2),
        #     keras.layers.Dense(4096, activation='relu'),
        #     #keras.layers.Dropout(0.2),
        #     keras.layers.Dense(1, activation='linear')
        # ])

        self.model = keras.models.Sequential(
            [
                keras.layers.InputLayer(input_shape=(65, 65, 1)),
                # keras.layers.BatchNormalization(),
                # keras.layers.GaussianNoise(0., name='gauss'),
                keras.layers.Conv2D(
                    filters=16,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    activation="relu",
                    padding="valid",
                ),
                # keras.layers.BatchNormalization(),
                keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
                keras.layers.Conv2D(
                    filters=32,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    activation="relu",
                    padding="valid",
                ),
                # keras.layers.BatchNormalization(),
                keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
                keras.layers.Flatten(),
                keras.layers.Dense(64, activation="relu"),
                # keras.layers.Dropout(rate=0.1),
                keras.layers.Dense(32, activation="relu"),
                # keras.layers.Dropout(rate=0.1),
                keras.layers.Dense(4, activation="linear"),
            ]
        )

        # # define two sets of inputs
        # input_x = keras.Input(shape=(65,65,1))
        # input_y = keras.Input(shape=(65,65,1))
        # input_a = keras.Input(shape=(65,65,1))
        # input_b = keras.Input(shape=(65,65,1))
        # input_i = keras.Input(shape=(65,65,1))
        # # noise = 0.5
        # # the first branch operates on the first input
        # # x = keras.layers.GaussianNoise(noise, name='gauss_x')(input_x)
        # x = keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1),
        #                         activation='relu', padding="valid")(input_x)
        # x = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(x)
        # x = keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1),
        #                         activation='relu', padding="valid")(x)
        # x = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(x)
        # x = keras.layers.Flatten()(x)
        # x = keras.layers.Dense(64, activation="relu")(x)
        # x = keras.layers.Dense(32, activation="relu")(x)
        # x = keras.Model(inputs=input_x, outputs=x)
        # # the second branch opreates on the second input
        # # y = keras.layers.GaussianNoise(noise, name='gauss_y')(input_y)
        # y = keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1),
        #                         activation='relu', padding="valid")(input_y)
        # y = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(y)
        # y = keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1),
        #                         activation='relu', padding="valid")(y)
        # y = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(y)
        # y = keras.layers.Flatten()(y)
        # y = keras.layers.Dense(64, activation="relu")(y)
        # y = keras.layers.Dense(32, activation="relu")(y)
        # y = keras.Model(inputs=input_y, outputs=y)
        # # third branch
        # # a = keras.layers.GaussianNoise(noise, name='gauss_a')(input_a)
        # a = keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1),
        #                         activation='relu', padding="valid")(input_a)
        # a = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(a)
        # a = keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1),
        #                         activation='relu', padding="valid")(a)
        # a = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(a)
        # a = keras.layers.Flatten()(a)
        # a = keras.layers.Dense(64, activation="relu")(a)
        # a = keras.layers.Dense(32, activation="relu")(a)
        # a = keras.Model(inputs=input_a, outputs=a)
        # # fourth branch
        # # b = keras.layers.GaussianNoise(noise, name='gauss_b')(input_b)
        # b = keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1),
        #                         activation='relu', padding="valid")(input_b)
        # b = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(b)
        # b = keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1),
        #                         activation='relu', padding="valid")(b)
        # b = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(b)
        # b = keras.layers.Flatten()(b)
        # b = keras.layers.Dense(64, activation="relu")(b)
        # b = keras.layers.Dense(32, activation="relu")(b)
        # b = keras.Model(inputs=input_b, outputs=b)
        # #fith branch
        # i = keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1),
        #                         activation='relu', padding="valid")(input_i)
        # i = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(i)
        # i = keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1),
        #                         activation='relu', padding="valid")(i)
        # i = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(i)
        # i = keras.layers.Flatten()(i)
        # i = keras.layers.Dense(64, activation="relu")(i)
        # i = keras.layers.Dense(32, activation="relu")(i)
        # i = keras.Model(inputs=input_i, outputs=i)
        # # combine the output of the four branches
        # combined = keras.layers.concatenate([x.output, y.output, a.output, b.output,
        #                                       i.output])
        # # apply a FC layer and then a regression prediction on the
        # # combined outputs
        # z = keras.layers.Dense(128, activation="relu")(combined)
        # # z = keras.layers.Dense(64, activation="relu")(z)
        # # z = keras.layers.Dense(32, activation="relu")(z)
        # # z = keras.layers.Dense(16, activation="relu")(z)
        # z = keras.layers.Dense(4, activation="linear")(z)
        # # our model will accept the inputs of the two branches and
        # # then output a single value
        # self.model = keras.Model(inputs=[x.input, y.input, a.input, b.input,
        #                                   i.input], outputs=z)

        self.model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=["accuracy"],
        )

        # displaying all the model’s layers, including each layer’s
        # name, its output shape, and its number of parameters.
        # the summary ends with the total number of parameters, including
        # trainable and non-trainable parameters.
        print(self.model.summary())

    def train_model(self) -> None:
        """
        For training one simply needs to call its fit() method. Pass the input
        features (X_train) and the target classes (y_train), as well as the
        number of epochs to train (or else it would default to just 1, which
        would definitely not be enough to converge to a good solution).
        Also pass a validation set (this is optional): Keras will measure
        the loss and the extra metrics on this set at the end of each epoch.

        The ModelCheckpoint callback saves checkpoints of the model at regular
        intervals during training, by default at the end of each epoch.
        the EarlyStopping callback will interrupt training when it measures no
        progress on the validation set for a number of epochs (defined by the
        patience argument), and it will optionally rollback to the best model.
        Combine both callbacks to both save checkpoints of the model
        (in case the computer crashes), and actually interrupt training early
        when there is no more progress (to avoid wasting time and resources).
        """
        if not self.model:
            self.create_model()

        checkpoint_cb = keras.callbacks.ModelCheckpoint(
            self.model_name, monitor="val_loss", mode="min", save_best_only=True
        )

        early_stopping_cb = keras.callbacks.EarlyStopping(
            patience=20, restore_best_weights=True
        )

        # noise_change_cb = GaussianNoiseCallback()

        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.epochs,
            callbacks=[
                checkpoint_cb,
                early_stopping_cb,
                # noise_change_cb,
            ],
        )

        # save training history
        with open("history.pkl", "wb") as file:
            pickle.dump(self.history.history, file)

    def evaluate_model(self) -> None:
        """
        Evaluate model on the test set to estimate the generalization error
        """
        # generate an image of the model
        keras.utils.plot_model(self.model, to_file="model.png")

        print("\nEvaluating generalization:")
        self.model.evaluate(self.test_ds)

        cmap = plt.cm.coolwarm.copy()
        test = self.test_ds.take(1)
        for images, labels in test:
            for i in range(1):
                plt.title(
                    "True: "
                    + str(np.round(labels[i].numpy(), 3))
                    + "\nPred.: "
                    + str(
                        np.round(self.model.predict(test)[i], 3)
                    ),  # np.argmax for discrete director
                    fontsize=12,
                )
                plt.imshow(images[i].numpy(), cmap=cmap)
                plt.colorbar()
                plt.axis("off")
                plt.show()
                # plt.imshow(images[4][i].numpy(), cmap=cmap)
                # plt.colorbar()
                # plt.axis("off")
                # plt.show()

        # pred = self.model.predict(self.test_ds)
        # indices = []
        # wrong_pred = []
        # for images, labels in self.test_ds.take(-1):
        #     for j,v in enumerate(pred):
        #         #print(np.argmax(pred[j]))
        #         if (pred[j] - labels[j].numpy()) < -1.:
        #             indices.append(j)
        #             wrong_pred.append(np.argmax(pred[j]))
        # wrong_images_1 = [images[0][i].numpy() for i in indices]
        # wrong_images_2 = [images[1][i].numpy() for i in indices]
        # wrong_labels = [labels[i].numpy() for i in indices]

        # for i in range(len(wrong_images_1)):
        #     plt.title('True: '+str(wrong_labels[i])+
        #               ', Predicted: '+str(wrong_pred[i]),
        #               fontsize=12)
        #     plt.imshow(wrong_images_1[i], cmap=cmap)
        #     plt.colorbar()
        #     plt.axis("off")
        #     plt.show()
        #     plt.title('True: '+str(wrong_labels[i])+
        #               ', Predicted: '+str(wrong_pred[i]),
        #               fontsize=12)
        #     plt.imshow(wrong_images_2[i], cmap=cmap)
        #     plt.colorbar()
        #     plt.axis("off")
        #     plt.show()

        # with np.load('/home/obernauer/Studium/Physik/Masterarbeit/STM/Code/Plots'
        #              '/single.npz') as data:
        #     datax = data['DataX']
        # img_array = datax[tf.newaxis, ..., tf.newaxis] # Create a batch
        # img_array = tf.image.per_image_standardization(img_array)
        # # Resize images from 67x67 to 277x277
        # img_array = tf.image.resize(img_array, (227,227), 'nearest')

        # print(self.model.predict(img_array))

    def analyize_model(self) -> None:
        """
        Analyze accuracy of nematic model.
        """
        try:
            with np.load("error.npz") as data:
                error_img = data["DataX"].astype(np.float32)
                # label_pred = data['DataY'].astype(np.float32)
                test_size = data["DataZ"][0]
        except FileNotFoundError:
            print("Error file from image_error() needed.")
            test_size = 1000

        true = []
        predict = self.model.predict(self.test_ds.take(test_size))
        mse = []
        error = []
        counter = 0

        for images, labels in self.test_ds.take(test_size):
            true.append(labels.numpy())
            mse.append(
                np.square(np.subtract(labels.numpy(), predict[counter])).mean()
                * np.array([1] * len(true[0]))
            )
            error.append(np.abs(np.subtract(labels.numpy(), predict[counter])))
            counter += 1

        true = np.array([item for sublist in true for item in sublist])
        mse = np.array([item for sublist in mse for item in sublist])
        error = np.array([item for sublist in error for item in sublist])

        for i in range(0, 4):
            plt.clf()
            plt.tick_params(which="both", width=2.5, direction="in")
            plt.tick_params(which="major", length=7)
            plt.tick_params(which="minor", length=4)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.minorticks_on()
            font_size = 16  # Adjust as appropriate.
            font_sizemae = 14  # Adjust as appropriate.
            plt.scatter(true[:, i], predict[:, i], c=error[:, i], cmap="plasma")
            plt.plot([0.01, 0.1], [0.01, 0.1], "w--")
            plt.xlabel(r"$\alpha_" + str(i) + "$ true", fontsize=font_size)
            plt.ylabel(r"$\alpha_" + str(i) + "$ predicted", fontsize=font_size)
            cbar = plt.colorbar(format=OOMFormatter(-3, mathText=True))
            cbar.set_label("MAE", rotation=270, labelpad=20, fontsize=font_sizemae)
            cbar.ax.tick_params(labelsize=font_sizemae)
            mape, rsq = stat(true[:, i], predict[:, i])
            metrics = f"$R^{2}$: " + str(rsq) + "\n" + "MAPE: " + str(mape)
            plt.text(
                0.09,
                0.02,
                metrics,
                bbox=dict(facecolor="none", edgecolor="black", boxstyle="round,pad=1"),
                fontsize=font_sizemae,
                ha="right",
            )
            plt.tight_layout()
            plt.savefig(f"alpha{i}.pdf")

        # for i in range(1, 4):
        #     plt.scatter(true[:, 0], true[:, i], c=((error[:, 0]**2+error[:, i]**2)/2.), cmap='plasma')
        #     plt.xlabel(r'$\alpha_'+str(0)+'$ true')
        #     plt.ylabel(r'$\alpha_'+str(i)+'$ true')
        #     cbar = plt.colorbar()
        #     cbar.set_label('MSE', rotation=270, labelpad=15)
        #     plt.grid()
        #     plt.show()

        # plt.scatter(true[:, 1], true[:, 2], c=((error[:, 1]**2+error[:, 2]**2)/2.), cmap='plasma')
        # plt.xlabel(r'$\alpha_1$ true')
        # plt.ylabel(r'$\alpha_2$ true')
        # cbar = plt.colorbar()
        # cbar.set_label('MSE', rotation=270, labelpad=15)
        # plt.grid()
        # plt.show()

        # plt.scatter(true[:, 1], true[:, 3], c=((error[:, 1]**2+error[:, 3]**2)/2.), cmap='plasma')
        # plt.xlabel(r'$\alpha_1$ true')
        # plt.ylabel(r'$\alpha_3$ true')
        # cbar = plt.colorbar()
        # cbar.set_label('MSE', rotation=270, labelpad=15)
        # plt.grid()
        # plt.show()

        # plt.scatter(true[:, 2], true[:, 3], c=((error[:, 2]**2+error[:, 3]**2)/2.), cmap='plasma')
        # plt.xlabel(r'$\alpha_2$ true')
        # plt.ylabel(r'$\alpha_3$ true')
        # cbar = plt.colorbar()
        # cbar.set_label('MSE', rotation=270, labelpad=15)
        # plt.grid()
        # plt.show()

        # plt.scatter(true, predict, c=mse, cmap='plasma')
        # plt.plot([0.0, np.pi], [0.0, np.pi], 'r--')
        # plt.xlabel(r'$\theta$ true')
        # plt.ylabel(r'$\theta$ predicted')
        # cbar = plt.colorbar()
        # cbar.set_label('MSE', rotation=270, labelpad=15)
        # plt.grid()
        # plt.show()

    def image_error(self) -> None:
        """
        Calculates an image where each pixel is the difference between
        |I(alpha_true)|-|I(alpha_predicted)| averaged over the four energies.
        Then the maximum error pixel is taken.
        """
        # setup model to generate I(alpha_predicted)
        self.full = True
        self.nematic = True
        self.bravais_lattice(7, False)
        self.k_karman(50)
        self.eta = 0.01
        self.theta = 5 * np.pi / 6.0

        size = 500
        error = []
        max_err = []
        alpha_pred = []
        for test_images, labels in tqdm(
            self.test_ds.take(size), position=0, leave=True
        ):
            alpha_pred.append(self.model.predict(test_images)[0])
            self.alpha = alpha_pred[-1]
            error_single = []
            for count, w in enumerate([-2, -1, 1, 2]):
                self.frequency = w
                g_arr = self.green()
                didv = self.didv(g_arr)
                test = np.reshape(test_images[count], (65, 65))
                error_single.append(np.array(np.abs(didv[:, ::-1].T - test)))
            error.append((np.mean(error_single, axis=0)))
            max_err.append(np.amax(error[-1]))
        # plt.imshow(error[0])
        # plt.colorbar()
        np.savez(
            "error.npz",
            DataX=max_err,
            DataY=alpha_pred,
            DataZ=(size * np.array([1] * len(max_err))),
        )


def main_train(model: str, batch_size: int, epochs: int) -> None:
    """
    Main function for training the neural network
    """
    ml = ML(model, batch_size, epochs)

    # ml.process_training_data()
    # ml.plot_training_data()
    # # #ml.create_model()
    # ml.train_model()
    # plot_history()

    ml.process_test_data()
    # ml.evaluate_model()
    # ml.image_error()
    ml.analyize_model()


if __name__ == "__main__":
    cfg = {
        # "model": "model.h5",
        "model": "appd_model.h5",
        "batch_size": 32,
        "epochs": 20000,
    }

    main_train(
        cfg["model"],
        cfg["batch_size"],
        cfg["epochs"],
    )
