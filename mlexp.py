#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 10:04:52 2022

@author: obernauer

https://towardsdatascience.com/techniques-for-handling-underfitting-and-overfitting-in-machine-learning-348daa2380b9
Pre processing: Def process_images 65x65 npz files.
Batch normalization: 
https://twitter.com/aureliengeron/status/1110839223878184960
https://towardsdatascience.com/why-batchnorm-works-518bb004bc58
https://towardsdatascience.com/batch-norm-explained-visually-how-it-works-and-why-neural-networks-need-it-b18919692739
Training data: 

https://pyimagesearch.com/2021/05/14/convolutional-neural-networks-cnns-and-layer-types/
References
[1] Crash course on CNN's:
    -https://pyimagesearch.com/2021/05/14/convolutional-neural-networks-cnns
    -and-layer-types/
About validation: 
https://stackoverflow.com/questions/48226086/training-loss-and-validation-loss-in-deep-learning
"""
import os
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
from matplotlib import rc
import matplotlib
import math

rc("font", family="serif", serif="cm10")
# rc('text', usetex=True)
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = [
    r"\usepackage{amsmath}"
]  # for \text commandatplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']
# Plot label font configuration
# rc('font',**{'family':'serif','serif':['Helvetica']})
# rc('text', usetex=True)

# font = {'family' : 'normal',
#        'weight' : 'bold',
#        'size'   : 45}


def process_images(image):
    """
    Add additional channel axis to image. Convert image from float64 to float32.
    Add gaussian noise.
    https://towardsdatascience.com/implementing-alexnet-cnn-architecture-using-
    tensorflow-2-0-and-keras-2113e090ad98?gi=164148c31297
    """
    # image = image / 100.
    image = image[..., tf.newaxis]  # [height, width, channels]
    image = tf.image.convert_image_dtype(image, tf.float32)
    batch, row, col, ch = image.shape
    mean = 0
    sigma = 0.31
    # sigma = 1
    gauss = np.abs(np.random.normal(mean, sigma, (batch, row, col, ch)))
    gauss = gauss.reshape(batch, row, col, ch)
    image = image + gauss
    # image = image + gauss
    return image


def process_images2(image):
    """
    Add additional channel axis to image. Convert image from float64 to float32.
    Add gaussian noise.
    https://towardsdatascience.com/implementing-alexnet-cnn-architecture-using-
    tensorflow-2-0-and-keras-2113e090ad98?gi=164148c31297
    """
    # image = image / 100.
    image = image[..., tf.newaxis]  # [height, width, channels]
    image = tf.image.convert_image_dtype(image, tf.float32)
    batch, row, col, ch = image.shape
    mean = 0
    sigma = 0
    gauss = np.abs(np.random.normal(mean, sigma, (batch, row, col, ch)))
    gauss = gauss.reshape(batch, row, col, ch)
    image = image + gauss
    # image = image + gauss
    return image


def process_images(image):
    """
    Add additional channel axis to image. Convert image from float64 to float32.
    Add gaussian noise.
    https://towardsdatascience.com/implementing-alexnet-cnn-architecture-using-
    tensorflow-2-0-and-keras-2113e090ad98?gi=164148c31297
    """
    # image = image / 100.
    image = image[..., tf.newaxis]  # [height, width, channels]
    image = tf.image.convert_image_dtype(image, tf.float32)
    batch, row, col, ch = image.shape
    mean = 0
    # sigma = 0.31
    # sigma = 1
    sigma = 0
    gauss = np.abs(np.random.normal(mean, sigma, (batch, row, col, ch)))
    gauss = gauss.reshape(batch, row, col, ch)
    image = image + gauss
    # image = image + gauss
    return image


# def get_gaussian_noise_output(model, epoch) -> None:
#     """
#     Plotting the output of a gaussian noise layer added with
#     keras.layers.GaussianNoise.
#
#     Parameters
#     ----------
#     model :
#         Model with a gaussian noise layer.
#     epoch : int
#         count for number of trained epochs.
#     """
#     with np.load("training.npz") as data:
#         image1 = data["DataX"][0]
#         image2 = data["DataX"][1]
#         image3 = data["DataX"][2]
#         image4 = data["DataX"][3]
#         label = data["DataY"][0]
#
#     partial_model = keras.Model(model.inputs, model.get_layer("gauss_x").output)
#
#     output_train = partial_model((image1, image2, image3, image4), training=True)
#     # runs the model in training mode
#
#     cmap = plt.cm.coolwarm.copy()
#     if epoch == 0:
#         plt.imshow(image1, cmap=cmap)
#         plt.title(
#             "Original with " r"$\alpha = $" + str(np.round(label, 3)), fontsize=10
#         )
#         plt.colorbar()
#         plt.axis("off")
#         plt.show()
#     plt.imshow(output_train, cmap=cmap)
#     plt.title(
#         "Added Gaussian noise with "
#         r"$\sigma = $" + str(np.round(model.get_layer("gauss_x").stddev, 2)),
#         fontsize=10,
#     )
#     plt.colorbar()
#     plt.axis("off")
#     plt.show()
#


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
    start = 1  # at which epoch the plot should start
    plt.minorticks_on()
    plt.tick_params(which="both", width=0.7, direction="in")
    plt.plot(range(start, len(loss)), loss[start:], "r", label="Training")
    plt.plot(range(start, len(val_loss)), val_loss[start:], "b", label="Validation")
    plt.legend()
    # plt.grid(True)
    plt.xlabel(r"Epochs", fontsize="large")
    plt.ylabel(r"Loss", fontsize="large")
    plt.savefig("loss.pdf")
    # plt.show()
    plt.minorticks_on()
    plt.tick_params(which="both", width=0.7, direction="in")
    plt.plot(range(1, len(acc) + 1), acc, "r", label="Train")
    plt.plot(range(1, len(val_acc) + 1), val_acc, "b", label="Validation")
    plt.legend(loc="upper right", prop={"size": 11}, markerscale=5)
    plt.legend()
    # plt.grid(True)
    plt.xlabel(r"Epochs", fontsize="large")
    plt.ylabel(r"MAE", fontsize="large")
    plt.savefig("strainmae.pdf")
    # plt.show()


# class GaussianNoiseCallback(tf.keras.callbacks.Callback):
#     """
#     Class for adjusting the standard dev of the gaussian layer after every epoch.
#     """
#
#     def on_epoch_begin(self, epoch, logs=None):
#         """
#         Adjusting the std dev and plotting an example via get_gaussian_noise_output()
#
#         Parameters
#         ----------
#         Are needed due to class inheritance.
#         """
#         # stddev = random.uniform(0, 1)
#         self.model.get_layer("gauss_x").stddev = random.uniform(0, 1)
#         self.model.get_layer("gauss_y").stddev = random.uniform(0, 1)
#         self.model.get_layer("gauss_a").stddev = random.uniform(0, 1)
#         self.model.get_layer("gauss_b").stddev = random.uniform(0, 1)
#         get_gaussian_noise_output(self.model, epoch)


class ML:
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
        Either a loaded model with model_name from h5 file or a newly generated one.
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
        Define a model, the optimizer and loss function.
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

        # JA: call init from model class
        super().__init__()

    def process_training_data(self):
        """
        Reading in and processing training and validation data from
        npz files on hard disk.
        """
        # reading in scaleograms. These images are not processed with process_images
        # because the noise was already added at the ldos generation
        # with np.load("trafo_train.npz") as data:
        #     trafo_train = data["DataX"].astype(np.float32)[..., tf.newaxis]
        # JA: Adding a new axis at the end of the tensor [..., tf.newaxis]
        # same as  expanded_1 = tf.expand_dims(trafo_train,axis=1)
        # astype converts every value to float32.

        # reading in stm images
        # There are 6 thousand images for each energy in (-15,-10,-5,1) = 2400
        # Two parameters for each figure
        nlabels = 8000
        with np.load("/scratch/c7051184/Trainingstrainangles.npz") as data:
            # JA:seq[start:end:step]
            # Each has a different energy (4 total). All the same alpha
            train_examples_1 = data["DataZ"][0 : 3 * nlabels : 3]
            train_examples_2 = data["DataZ"][1 : 3 * nlabels + 1 : 3]
            train_examples_3 = data["DataZ"][2 : 3 * nlabels + 2 : 3]
            train_examples_4 = data["DataX"][0 : 4 * nlabels : 4]
            train_examples_5 = data["DataX"][1 : 4 * nlabels + 1 : 4]
            # train_examples_6 = data["DataX"][2 : 4 * nlabels + 2 : 4]
            # train_examples_7 = data["DataX"][3 : 4 * nlabels + 3 : 4]
            train_labels = data["DataY"].astype(np.float32)[0:nlabels]

            # train_examples_1 = data["DataX"][0 : 4 * nlabels : 4]
            # train_examples_2 = data["DataX"][1 : 4 * nlabels + 1 : 4]
            # train_examples_3 = data["DataX"][2 : 4 * nlabels + 2 : 4]
            # train_examples_4 = data["DataX"][3 : 4 * nlabels + 3 : 4]
            # trafo_train = data["DataZ"][4 : 5 * nlabels + 4 : 5]
            # train_labels = data["DataY"].astype(np.float32)[0:nlabels]

        # print(train_examples_1.shape)
        # print(train_examples_2.shape)
        # print(train_examples_3.shape)
        # print(train_examples_4.shape)
        # print(trafo_train.shape)
        print(train_labels.shape)
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (
                (
                    process_images(train_examples_1),
                    process_images(train_examples_2),
                    process_images(train_examples_3),
                    process_images(train_examples_4),
                    process_images(train_examples_5),
                    # process_images(train_examples_6),
                    # process_images(train_examples_7),
                    # process_images(train_examples_4),
                    # process_images(trafo_train),
                ),
                train_labels,
            )
        )
        #  1st
        print(train_dataset)
        train_ds_size = tf.data.experimental.cardinality(train_dataset).numpy()
        print("Training data size of one energy channel:", train_ds_size)
        self.train_ds = (
            train_dataset.shuffle(buffer_size=train_ds_size)
            .batch(batch_size=self.batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )
        nlabels = 2000
        with np.load("/scratch/c7051184/Validationstrainangles.npz") as data:
            val_examples_1 = data["DataZ"][0 : 3 * nlabels : 3]
            val_examples_2 = data["DataZ"][1 : 3 * nlabels + 1 : 3]
            val_examples_3 = data["DataZ"][2 : 3 * nlabels + 2 : 3]
            val_examples_4 = data["DataX"][0 : 4 * nlabels : 4]
            val_examples_5 = data["DataX"][1 : 4 * nlabels + 1 : 4]
            # val_examples_6 = data["DataX"][2 : 4 * nlabels + 2: 4]
            # val_examples_7 = data["DataX"][3 : 4 * nlabels + 3 : 4]
            val_labels = data["DataY"].astype(np.float32)[0:nlabels]
            # val_examples_1 = data["DataX"][0 : 4 * nlabels : 4]
            # val_examples_2 = data["DataX"][1 : 4 * nlabels + 1 : 4]
            # val_examples_3 = data["DataX"][2 : 4 * nlabels + 2 : 4]
            # val_examples_4 = data["DataX"][3 : 4 * nlabels + 3 : 4]
            # trafo_val = data["DataZ"][3 : 5 * nlabels + 3 : 5]
            # val_labels = data["DataY"].astype(np.float32)[0:nlabels]

        val_dataset = tf.data.Dataset.from_tensor_slices(
            (
                (
                    process_images(val_examples_1),
                    process_images(val_examples_2),
                    process_images(val_examples_3),
                    process_images(val_examples_4),
                    process_images(val_examples_5),
                    # process_images(val_examples_6),
                    # process_images(val_examples_7),
                    # process_images(trafo_val),
                ),
                val_labels,
            )
        )
        val_ds_size = tf.data.experimental.cardinality(val_dataset).numpy()
        print(val_dataset)
        print("Validation data size of one energy channel:", val_ds_size)
        self.val_ds = (
            val_dataset.shuffle(buffer_size=val_ds_size)
            .batch(batch_size=self.batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )
        #

    def process_test_data(self) -> None:
        """
        Reading in and processing training and validation data from
        npz files on hard disk.
        """
        # with np.load("trafo_test.npz") as data:
        #     trafo_test = data["DataX"].astype(np.float32)[..., tf.newaxis]
        #
        nlabels = 2000
        with np.load("/scratch/c7051184/Teststrainanglesn.npz") as data:
            test_examples_1 = data["DataZ"][0 : 3 * nlabels : 3]
            test_examples_2 = data["DataZ"][1 : 3 * nlabels + 1 : 3]
            test_examples_3 = data["DataZ"][2 : 3 * nlabels + 2 : 3]
            test_examples_4 = data["DataX"][0 : 4 * nlabels : 4]
            test_examples_5 = data["DataX"][1 : 4 * nlabels + 1 : 4]
            # test_examples_6 = data["DataX"][2 : 4 * nlabels + 2: 4]
            # test_examples_7 = data["DataX"][3 : 4 * nlabels + 3: 4]
            test_labels = data["DataY"].astype(np.float32)[0:nlabels]
            test_labels2 = data["DataA"].astype(np.float32)[0:nlabels]

        test_dataset = tf.data.Dataset.from_tensor_slices(
            (
                (
                    process_images(test_examples_1),
                    process_images(test_examples_2),
                    process_images(test_examples_3),
                    process_images(test_examples_4),
                    process_images(test_examples_5),
                    # process_images(test_examples_6),
                    # process_images(test_examples_7),
                ),
                test_labels,
            )
        )

        test_dataset2 = tf.data.Dataset.from_tensor_slices(
            (
                (
                    process_images(test_examples_1),
                    process_images(test_examples_2),
                    process_images(test_examples_3),
                    process_images(test_examples_4),
                    process_images(test_examples_5),
                    # process_images(test_examples_6),
                    # process_images(test_examples_7),
                ),
                test_labels2,
            )
        )
        test_ds_size = tf.data.experimental.cardinality(test_dataset).numpy()

        test_ds_size2 = tf.data.experimental.cardinality(test_dataset2).numpy()
        print(test_dataset)
        print("Test data size of one energy channel:", test_ds_size)
        self.test_ds = test_dataset.batch(batch_size=1, drop_remainder=True)
        print(self.test_ds)
        self.test_ds2 = test_dataset2.batch(batch_size=1, drop_remainder=True)

    def process_training_data_7channels(self):
        nlabels = 8000
        with np.load("/scratch/c7051184/Trainingstrainangles.npz") as data:
            # JA:seq[start:end:step]
            # Each has a different energy (4 total). All the same alpha
            train_examples_1 = data["DataW"][0 : 3 * nlabels : 3]
            train_examples_2 = data["DataW"][1 : 3 * nlabels + 1 : 3]
            train_examples_3 = data["DataW"][2 : 3 * nlabels + 2 : 3]
            train_examples_4 = data["DataX"][0 : 4 * nlabels : 4]
            train_examples_5 = data["DataX"][1 : 4 * nlabels + 1 : 4]
            train_examples_6 = data["DataX"][2 : 4 * nlabels + 2 : 4]
            train_examples_7 = data["DataX"][3 : 4 * nlabels + 3 : 4]
            train_labels = data["DataY"].astype(np.float32)[0:nlabels]

        print(train_labels.shape)
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (
                (
                    process_images2(train_examples_1),
                    process_images2(train_examples_2),
                    process_images2(train_examples_3),
                    process_images(train_examples_4),
                    process_images(train_examples_5),
                    process_images(train_examples_6),
                    process_images(train_examples_7),
                ),
                train_labels,
            )
        )
        #  1st
        print(train_dataset)
        train_ds_size = tf.data.experimental.cardinality(train_dataset).numpy()
        print("Training data size of one energy channel:", train_ds_size)
        self.train_ds = (
            train_dataset.shuffle(buffer_size=train_ds_size)
            .batch(batch_size=self.batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )
        nlabels = 2000
        with np.load("/scratch/c7051184/Validationstrainangles.npz") as data:
            val_examples_1 = data["DataW"][0 : 3 * nlabels : 3]
            val_examples_2 = data["DataW"][1 : 3 * nlabels + 1 : 3]
            val_examples_3 = data["DataW"][2 : 3 * nlabels + 2 : 3]
            val_examples_4 = data["DataX"][0 : 4 * nlabels : 4]
            val_examples_5 = data["DataX"][1 : 4 * nlabels + 1 : 4]
            val_examples_6 = data["DataX"][2 : 4 * nlabels + 2 : 4]
            val_examples_7 = data["DataX"][3 : 4 * nlabels + 3 : 4]
            val_labels = data["DataY"].astype(np.float32)[0:nlabels]

        val_dataset = tf.data.Dataset.from_tensor_slices(
            (
                (
                    process_images2(val_examples_1),
                    process_images2(val_examples_2),
                    process_images2(val_examples_3),
                    process_images(val_examples_4),
                    process_images(val_examples_5),
                    process_images(val_examples_6),
                    process_images(val_examples_7),
                ),
                val_labels,
            )
        )
        val_ds_size = tf.data.experimental.cardinality(val_dataset).numpy()
        print(val_dataset)
        print("Validation data size of one energy channel:", val_ds_size)
        self.val_ds = (
            val_dataset.shuffle(buffer_size=val_ds_size)
            .batch(batch_size=self.batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )
        #

    def process_test_data_5channels(self) -> None:
        """
        Reading in and processing training and validation data from
        npz files on hard disk.
        """
        # with np.load("trafo_test.npz") as data:
        #     trafo_test = data["DataX"].astype(np.float32)[..., tf.newaxis]
        #
        nlabels = 7
        with np.load(
            "/home/jass/Downloads/datasetstrain/same padding/expdata/expdata.npz"
        ) as data:
            # test_examples_1 = data["DataZ"][0 : 3 * nlabels : 3]
            # test_examples_2 = data["DataZ"][1 : 3 * nlabels + 1 : 3]
            test_examples_3 = data["DataZ"][2 : 3 * nlabels + 2 : 3]
            test_examples_4 = data["DataX"][0 : 4 * nlabels : 4]
            test_examples_5 = data["DataX"][1 : 4 * nlabels + 1 : 4]
            test_examples_6 = data["DataX"][2 : 4 * nlabels + 2 : 4]
            test_examples_7 = data["DataX"][3 : 4 * nlabels + 3 : 4]
            test_labels = data["DataP"].astype(np.float32)[0:nlabels]

        # print(test_examples_1.shape)
        # print(test_examples_4.shape)
        test_dataset = tf.data.Dataset.from_tensor_slices(
            (
                (
                    # process_images2(test_examples_1),
                    # process_images2(test_examples_2),
                    process_images2(test_examples_3),
                    process_images2(test_examples_4),
                    process_images2(test_examples_5),
                    process_images2(test_examples_6),
                    process_images2(test_examples_7),
                ),
                test_labels,
            )
        )

        test_ds_size = tf.data.experimental.cardinality(test_dataset).numpy()

        print(test_dataset)
        print("Test data size of one energy channel:", test_ds_size)
        self.test_ds = test_dataset.batch(batch_size=1, drop_remainder=True)
        print(self.test_ds)
    def process_test_data_7channels(self) -> None:
        """
        Reading in and processing training and validation data from
        npz files on hard disk.
        """
        # with np.load("trafo_test.npz") as data:
        #     trafo_test = data["DataX"].astype(np.float32)[..., tf.newaxis]
        #
        nlabels = 7
        with np.load(
            "/home/jass/Downloads/datasetstrain/same padding/expdata/expdata.npz"
        ) as data:
            test_examples_1 = data["DataZ"][0 : 3 * nlabels : 3]
            test_examples_2 = data["DataZ"][1 : 3 * nlabels + 1 : 3]
            test_examples_3 = data["DataZ"][2 : 3 * nlabels + 2 : 3]
            test_examples_4 = data["DataX"][0 : 4 * nlabels : 4]
            test_examples_5 = data["DataX"][1 : 4 * nlabels + 1 : 4]
            test_examples_6 = data["DataX"][2 : 4 * nlabels + 2 : 4]
            test_examples_7 = data["DataX"][3 : 4 * nlabels + 3 : 4]
            test_labels = data["DataP"].astype(np.float32)[0:nlabels]

        # print(test_examples_1.shape)
        # print(test_examples_4.shape)
        test_dataset = tf.data.Dataset.from_tensor_slices(
            (
                (
                    process_images2(test_examples_1),
                    process_images2(test_examples_2),
                    process_images2(test_examples_3),
                    process_images2(test_examples_4),
                    process_images2(test_examples_5),
                    process_images2(test_examples_6),
                    process_images2(test_examples_7),
                ),
                test_labels,
            )
        )

        test_ds_size = tf.data.experimental.cardinality(test_dataset).numpy()

        print(test_dataset)
        print("Test data size of one energy channel:", test_ds_size)
        self.test_ds = test_dataset.batch(batch_size=1, drop_remainder=True)
        print(self.test_ds)

    def plot_training_data(self) -> None:
        """
        Plotting examples of the training data.
        """
        cmap = plt.cm.coolwarm.copy()
        for images, labels in self.train_ds.take(1):
            for i in range(0, 100):
                plt.title(np.round(labels[i].numpy(), 4), fontsize=12)
                plt.imshow(images[0][i].numpy(), cmap=cmap)
                plt.colorbar()
                plt.axis("off")
                plt.show()
                # plt.title(np.round(labels[0][1:3].numpy(), 4), fontsize=12)
                # plt.imshow(images[0].numpy(), cmap=cmap)
                # plt.colorbar()
                # plt.axis("off")
                # plt.show()

    def create_model_7channels(self) -> None:
        # # model I used for 4 + 1 channels
        input_x = keras.Input(shape=(65, 65, 1))
        input_y = keras.Input(shape=(65, 65, 1))
        input_a = keras.Input(shape=(65, 65, 1))
        input_b = keras.Input(shape=(65, 65, 1))
        input_i = keras.Input(shape=(65, 65, 1))
        input_p = keras.Input(shape=(65, 65, 1))
        input_q = keras.Input(shape=(65, 65, 1))
        # # the first branch operates on the first input
        x = keras.layers.BatchNormalization()(input_x)
        x = keras.layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = keras.layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(64, activation="relu")(x)
        x = keras.layers.Dense(32, activation="relu")(x)
        x = keras.layers.Dense(16, activation="relu")(x)
        x = keras.Model(inputs=input_x, outputs=x)
        # the second branch opreates on the second input
        y = keras.layers.BatchNormalization()(input_y)
        y = keras.layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(y)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(y)
        y = keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(y)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(y)
        y = keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(y)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(y)
        y = keras.layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(y)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(y)
        y = keras.layers.Flatten()(y)
        y = keras.layers.Dense(64, activation="relu")(y)
        y = keras.layers.Dense(32, activation="relu")(y)
        y = keras.layers.Dense(16, activation="relu")(y)
        y = keras.Model(inputs=input_y, outputs=y)
        # third branch
        a = keras.layers.BatchNormalization()(input_a)
        a = keras.layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(a)
        a = keras.layers.BatchNormalization()(a)
        a = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(a)
        a = keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(a)
        a = keras.layers.BatchNormalization()(a)
        a = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(a)
        a = keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(a)
        a = keras.layers.BatchNormalization()(a)
        a = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(a)
        a = keras.layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(a)
        a = keras.layers.BatchNormalization()(a)
        a = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(a)
        a = keras.layers.Flatten()(a)
        a = keras.layers.Dense(64, activation="relu")(a)
        a = keras.layers.Dense(32, activation="relu")(a)
        a = keras.layers.Dense(16, activation="relu")(a)
        a = keras.Model(inputs=input_a, outputs=a)
        # forth branch
        b = keras.layers.BatchNormalization()(input_b)
        b = keras.layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(b)
        b = keras.layers.BatchNormalization()(b)
        b = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(b)
        b = keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(b)
        b = keras.layers.BatchNormalization()(b)
        b = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(b)
        b = keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(b)
        b = keras.layers.BatchNormalization()(b)
        b = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(b)
        b = keras.layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(b)
        b = keras.layers.BatchNormalization()(b)
        b = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(b)
        b = keras.layers.Flatten()(b)
        b = keras.layers.Dense(64, activation="relu")(b)
        b = keras.layers.Dense(32, activation="relu")(b)
        b = keras.layers.Dense(16, activation="relu")(b)
        b = keras.Model(inputs=input_b, outputs=b)
        # # # fith branch
        i = keras.layers.BatchNormalization()(input_i)
        i = keras.layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(i)
        i = keras.layers.BatchNormalization()(i)
        i = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(i)
        i = keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(i)
        i = keras.layers.BatchNormalization()(i)
        i = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(i)
        i = keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(i)
        i = keras.layers.BatchNormalization()(i)
        i = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(i)
        i = keras.layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(i)
        i = keras.layers.BatchNormalization()(i)
        i = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(i)
        i = keras.layers.Flatten()(i)
        i = keras.layers.Dense(64, activation="relu")(i)
        i = keras.layers.Dense(32, activation="relu")(i)
        i = keras.layers.Dense(16, activation="relu")(i)
        i = keras.Model(inputs=input_i, outputs=i)
        p = keras.layers.BatchNormalization()(input_p)
        p = keras.layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(p)
        p = keras.layers.BatchNormalization()(p)
        p = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(p)
        p = keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(p)
        p = keras.layers.BatchNormalization()(p)
        p = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(p)
        p = keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(p)
        p = keras.layers.BatchNormalization()(p)
        p = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(p)
        p = keras.layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(p)
        p = keras.layers.BatchNormalization()(p)
        p = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(p)
        p = keras.layers.Flatten()(p)
        p = keras.layers.Dense(64, activation="relu")(p)
        p = keras.layers.Dense(32, activation="relu")(p)
        p = keras.layers.Dense(16, activation="relu")(p)
        p = keras.Model(inputs=input_p, outputs=p)
        q = keras.layers.BatchNormalization()(input_q)
        q = keras.layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(q)
        q = keras.layers.BatchNormalization()(q)
        q = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(q)
        q = keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(q)
        q = keras.layers.BatchNormalization()(q)
        q = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(q)
        q = keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(q)
        q = keras.layers.BatchNormalization()(q)
        q = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(q)
        q = keras.layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(q)
        q = keras.layers.BatchNormalization()(q)
        q = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(q)
        q = keras.layers.Flatten()(q)
        q = keras.layers.Dense(64, activation="relu")(q)
        q = keras.layers.Dense(32, activation="relu")(q)
        q = keras.layers.Dense(16, activation="relu")(q)
        q = keras.Model(inputs=input_q, outputs=q)
        combined = keras.layers.concatenate(
            [x.output, y.output, a.output, b.output, i.output, p.output, q.output]
        )
        # # # apply a FC layer and then a regression prediction on the
        # # # combined outputs
        z = keras.layers.Dense(4096, activation="relu")(combined)
        z = keras.layers.Dropout(0.2)(z)
        # z = keras.layers.Dense(64, activation="relu")(z)
        # z = keras.layers.Dense(32, activation="relu")(z)
        # z = keras.layers.Dense(16, activation="relu")(z)
        # z = keras.layers.Dropout(rate=0.1)(z)
        z = keras.layers.Dense(5, activation="linear")(z)
        # # # our model will accept the inputs of the branches and
        # # # then output a single value
        self.model = keras.Model(
            inputs=[x.input, y.input, a.input, b.input, i.input, p.input, q.input],
            outputs=z,
        )
        # #
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=["mae"],
        )

        # displaying all the model’s layers, including each layer’s
        # name, its output shape, and its number of parameters.
        # the summary ends with the total number of parameters, including
        # trainable and non-trainable parameters.
        print(self.model.summary())

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
        Dense layer: here it will use the relu activation function.
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
        of extra metrics to compute during training and evaluation.
        """
        # more complex model. I couldn't achive better results with this.
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

        # model I used for single channel
        # Output of conv layer (Width) = (Winput - kernelsize(1) + 2pad)/Stride)+1

        # Joao
        # Same for height. Padding = same + STRIDES = 1, output has same size
        # as input: e.g.: W = (65+ 3 - 3 + 2)/1 +1 = 65
        # Pooling layers appear between Conv2D to reduce the amount of
        # parameters. They also help control overfitting.
        # We choose non-overlapping pooling (F=2,S=2) goo to smaller spatial
        # dimensions (test Poll=3, s = 2 or even (F=2,S=1))

        # Test: Springenberg hypothesis: dont use pool layers

        # Batch normalization diminisheds # of epochs for NN training.
        # It also can help stabilize training, allowing larger variety of learning rates
        # and regularization strengths.
        # Generally provides lower finall loss + stable loss curve
        # To be added after the actvation relu. .THe reason is that we normalize
        # the positive valued features without statistically biasing them. This
        # second part is that if apply before the relu, we may turn some weights
        # to negative values which are clamped by nonlinear activiation functions
        # such as relu

        # Test batch normalization after and before the activation.
        # Test 1: Dropout - > Try between the dense layers with p=0.5. Reuce overfitting
        # by explicitly altering the network architecture at training time.
        #
        # It will ensure that multiple nodes will apply (instead of one), which
        # helps the model to be generalized.
        # Test 2: Also apply dropout for smaller probabilities p=0.10-0.25;
        # in earlier layers (after maxpooling)

        # Test 10: Try pINPUT => [[CONV => relu]*N => POOL?]*M => [FC => relu]*K => FC
        # with different N, M and K
        # Test 11: Try ALexNet or VGGNet: These deeper networks are usually good
        # when we have LOTS of training data + classification problem being
        # sufficiently challenging.

        # Stacking Conv before pools alow more complex features before the
        # destructive pooling is applied.

        # Test 12: Eremove all pools, use only CNN's and apply average pooling
        # at the end for input to softmax classifies: may lead to
        # less parameters + faster training time (See Googlenet Res Net and Squeeze Net )

        # Concatenate: GoogleLeNet stack multipled filters across channel dimensions
        # learning multi-level features.

        # General rules:
        # 1 - Square input layersfor better otimization of LINALG libraries;
        # 2 - Inputlayer/2^N after first conv: tweak filter + stride. This allows
        # the spatial inputs to be consistently down sampled via pool operations
        # in an efficient way
        # 3 - Conv Filters: (3x3, 5x5). 1X1 for advanced local features.
        # 7x7, 11x1 - first layer if images > 200x200 pixels;
        # - > After first filter, reduce filter size otherwish the spatial dimensions
        # of volume diminish too quickly

        # 4 -
        # 5 - Zero padding - if dont wan't to reduce spatial dimensions via
        # convolution. This applied to multipled stacking COV increases classification
        # accuracy. Keras does that automatically with "same"
        # 6 - Pool layers to reduce spatial dimensions then use the CNN's.
        # 7 - Fields > 3x3 in max pooling are very destructive, be careful.
        # 8 - Batch normalization slows the training time but usually makes
        # tuuning the hyperparameers easier by stabilizing the training. After
        # activation functions. Don't apply before the last dense layer.
        # 9 - Dropout between dense layers (50%). Between pool and Cov ~10-25%
        # # model I used for 4 + 1 channels
        input_x = keras.Input(shape=(65, 65, 1))
        input_y = keras.Input(shape=(65, 65, 1))
        input_a = keras.Input(shape=(65, 65, 1))
        input_b = keras.Input(shape=(65, 65, 1))
        input_i = keras.Input(shape=(65, 65, 1))
        # input_p = keras.Input(shape=(65, 65, 1))
        # input_q = keras.Input(shape=(65, 65, 1))
        # # the first branch operates on the first input
        x = keras.layers.BatchNormalization()(input_x)
        x = keras.layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = keras.layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(64, activation="relu")(x)
        x = keras.layers.Dense(32, activation="relu")(x)
        x = keras.layers.Dense(16, activation="relu")(x)
        x = keras.Model(inputs=input_x, outputs=x)
        # the second branch opreates on the second input
        y = keras.layers.BatchNormalization()(input_y)
        y = keras.layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(y)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(y)
        y = keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(y)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(y)
        y = keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(y)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(y)
        y = keras.layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(y)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(y)
        y = keras.layers.Flatten()(y)
        y = keras.layers.Dense(64, activation="relu")(y)
        y = keras.layers.Dense(32, activation="relu")(y)
        y = keras.layers.Dense(16, activation="relu")(y)
        y = keras.Model(inputs=input_y, outputs=y)
        # third branch
        a = keras.layers.BatchNormalization()(input_a)
        a = keras.layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(a)
        a = keras.layers.BatchNormalization()(a)
        a = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(a)
        a = keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(a)
        a = keras.layers.BatchNormalization()(a)
        a = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(a)
        a = keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(a)
        a = keras.layers.BatchNormalization()(a)
        a = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(a)
        a = keras.layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(a)
        a = keras.layers.BatchNormalization()(a)
        a = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(a)
        a = keras.layers.Flatten()(a)
        a = keras.layers.Dense(64, activation="relu")(a)
        a = keras.layers.Dense(32, activation="relu")(a)
        a = keras.layers.Dense(16, activation="relu")(a)
        a = keras.Model(inputs=input_a, outputs=a)
        # forth branch
        b = keras.layers.BatchNormalization()(input_b)
        b = keras.layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(b)
        b = keras.layers.BatchNormalization()(b)
        b = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(b)
        b = keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(b)
        b = keras.layers.BatchNormalization()(b)
        b = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(b)
        b = keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(b)
        b = keras.layers.BatchNormalization()(b)
        b = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(b)
        b = keras.layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(b)
        b = keras.layers.BatchNormalization()(b)
        b = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(b)
        b = keras.layers.Flatten()(b)
        b = keras.layers.Dense(64, activation="relu")(b)
        b = keras.layers.Dense(32, activation="relu")(b)
        b = keras.layers.Dense(16, activation="relu")(b)
        b = keras.Model(inputs=input_b, outputs=b)
        # # # fith branch
        i = keras.layers.BatchNormalization()(input_i)
        i = keras.layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(i)
        i = keras.layers.BatchNormalization()(i)
        i = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(i)
        i = keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(i)
        i = keras.layers.BatchNormalization()(i)
        i = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(i)
        i = keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(i)
        i = keras.layers.BatchNormalization()(i)
        i = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(i)
        i = keras.layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(i)
        i = keras.layers.BatchNormalization()(i)
        i = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(i)
        i = keras.layers.Flatten()(i)
        i = keras.layers.Dense(64, activation="relu")(i)
        i = keras.layers.Dense(32, activation="relu")(i)
        i = keras.layers.Dense(16, activation="relu")(i)
        i = keras.Model(inputs=input_i, outputs=i)
        combined = keras.layers.concatenate(
            [x.output, y.output, a.output, b.output, i.output]
        )
        # # # apply a FC layer and then a regression prediction on the
        # # # combined outputs
        z = keras.layers.Dense(4096, activation="relu")(combined)
        z = keras.layers.Dropout(0.2)(z)
        # z = keras.layers.Dense(64, activation="relu")(z)
        # z = keras.layers.Dense(32, activation="relu")(z)
        # z = keras.layers.Dense(16, activation="relu")(z)
        # z = keras.layers.Dropout(rate=0.1)(z)
        z = keras.layers.Dense(5, activation="linear")(z)
        # # # our model will accept the inputs of the branches and
        # # # then output a single value
        self.model = keras.Model(
            inputs=[x.input, y.input, a.input, b.input, i.input], outputs=z
        )
        # #
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=["mae"],
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
            patience=20000, restore_best_weights=True
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
        # keras.utils.plot_model(self.model, rankdir="LR", to_file="model.png")
        # keras.utils.plot_model(self.model, to_file="model3points.png")
        # print("\nEvaluating generalization:")
        # self.model.evaluate(self.test_ds)
        #
        # pcmap = plt.cm.inferno.copy()
        # plt.show()
        test = self.test_ds.take(7)
        for images, labels in test:
            for i in range(7):
                plt.title(
                    "True: "
                    + str(np.round(labels[0].numpy(), 3))
                    + "\nPred.: "
                    + str(np.round(self.model.predict(test)[i], 3)),
                    # np.argmax for discrete director
                    fontsize=12,
                )
                # plt.imshow(images[i].numpy(), cmap=cmap)
                # plt.colorbar()
                # plt.axis("off")
                # plt.show()
                # print(images[2][0])
                plt.imshow(images[i][0].numpy(), cmap="inferno")
                plt.colorbar()
                plt.axis("off")
                # plt.savefig("example.pdf")
                plt.show()

    def analyize_model_exp(self) -> None:
        """
        Analyze accuracy of nematic model.
        """

        test_size = 7
        true = []
        predict = self.model.predict(self.test_ds.take(test_size))
        counter = 0

        print(true, predict)
        for images, labels in self.test_ds.take(test_size):
            true.append(
                labels.numpy(),
            )
            counter += 1

        true = np.array([item for sublist in true for item in sublist])
        print(true)

        plt.clf()
        fig, ax = plt.subplots()
        # ax.set_aspect(1)
        plt.tick_params(which="both", width=1.5, direction="in")
        plt.tick_params(which="major", length=7)
        plt.tick_params(which="minor", length=4)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.minorticks_on()
        plt.tick_params(which="both", width=0.7, direction="in")
        plt.scatter(true[:], predict[:, 0])
        plt.xlabel(r"Filling of the CFB $(n_{s})$", size=15)
        plt.ylabel(r"$\Phi_{\mathrm{MN}}" + "$ predicted", size=15)
        plt.savefig(f"phimn_exp2.pdf")

        plt.clf()
        fig, ax = plt.subplots()
        # ax.set_aspect(1)
        plt.tick_params(which="both", width=1.5, direction="in")
        plt.tick_params(which="major", length=7)
        plt.tick_params(which="minor", length=4)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.minorticks_on()
        plt.tick_params(which="both", width=0.7, direction="in")
        plt.scatter(true[:], predict[:, 1])
        plt.xlabel(r"Filling of the CFB $(n_{s})$", size=15)
        plt.ylabel(r"$\Phi_{\mathrm{GN}}" + "$ predicted (eV)", size=15)
        plt.savefig(f"phign_exp2.pdf")

        predict[:, 2] = 100 * predict[:, 2]
        plt.clf()
        fig, ax = plt.subplots()
        # ax.set_aspect(1)
        plt.tick_params(which="both", width=1.5, direction="in")
        plt.tick_params(which="major", length=7)
        plt.tick_params(which="minor", length=4)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.minorticks_on()
        plt.tick_params(which="both", width=0.7, direction="in")
        plt.scatter(true[:], predict[:, 2])
        # plt.plot([-0.58, 0.67], [0, 0.8], linestyle="--", c="white")
        plt.xlabel(r"Filling of the CFB $(n_{s})$", size=15)
        plt.ylabel(r"$\epsilon" + "$ predicted ", size=15)
        plt.savefig(f"epsilon_exp2.pdf")

        # Prediction of angle
        true2 = []
        test2 = []
        counter = 0
        for images, labels in self.test_ds.take(test_size):
            true2.append(
                labels.numpy(),
            )
            cos = predict[counter][3]
            sin = predict[counter][4]
            temp = math.atan2(sin, cos)
            # print(temp)
            # #conversion to radian
            # temp = 180*temp/np.pi
            print(temp)
            test2.append(temp)
            counter += 1

        true2 = np.array([item for sublist in true2 for item in sublist])

        plt.clf()
        fig, ax = plt.subplots()
        # ax.set_aspect(1)
        plt.tick_params(which="both", width=1.5, direction="in")
        plt.tick_params(which="major", length=7)
        plt.tick_params(which="minor", length=4)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.minorticks_on()
        plt.scatter(true2, test2)
        plt.xlabel(r"Filling of the CFB $(n_{s})$", size=15)
        plt.ylabel(r"$\theta_{\varepsilon}" + "$ predicted", size=15)
        plt.tight_layout()
        plt.savefig(f"theta_exp2.pdf")

    def analyize_model(self) -> None:
        """
        Analyze accuracy of nematic model.
        """

        test_size = 2000
        true = []
        predict = self.model.predict(self.test_ds.take(test_size))
        mse = []
        error = []
        counter = 0

        for images, labels in self.test_ds.take(test_size):
            true.append(
                labels.numpy(),
            )
            mse.append(
                np.square(np.subtract(labels.numpy(), predict[counter])).mean()
                * np.array([1] * len(true[0]))
            )
            error.append(np.abs(np.subtract(labels.numpy(), predict[counter])))
            counter += 1

        true = np.array([item for sublist in true for item in sublist])
        mse = np.array([item for sublist in mse for item in sublist])
        error = np.array([item for sublist in error for item in sublist])

        error[:, 0] = 100 * error[:, 0]
        plt.clf()
        fig, ax = plt.subplots()
        # ax.set_aspect(1)
        plt.tick_params(which="both", width=1.5, direction="in")
        plt.tick_params(which="major", length=7)
        plt.tick_params(which="minor", length=4)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.minorticks_on()
        plt.tick_params(which="both", width=0.7, direction="in")
        plt.scatter(true[:, 0], predict[:, 0], c=error[:, 0], cmap="plasma")
        plt.plot([0, 0.1], [0, 0.1], linestyle="--", c="white")
        plt.xlabel(r"$\Phi_{\mathrm{MN}}" + "$ true", size=15)
        plt.ylabel(r"$\Phi_{\mathrm{MN}}" + "$ predicted", size=15)
        cbar = plt.colorbar()
        cbar.set_label("MAE", rotation=270, labelpad=15, size=15)
        cbar.ax.minorticks_on()
        cbar.ax.tick_params(labelsize=15)
        cbar.ax.tick_params(which="major", length=7)
        cbar.ax.tick_params(which="minor", length=4)
        cbar.ax.tick_params(which="both", width=1.5)
        plt.savefig(f"phimn.pdf")

        error[:, 1] = 100 * error[:, 1]
        plt.clf()
        fig, ax = plt.subplots()
        # ax.set_aspect(1)
        plt.tick_params(which="both", width=1.5, direction="in")
        plt.tick_params(which="major", length=7)
        plt.tick_params(which="minor", length=4)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.minorticks_on()
        plt.tick_params(which="both", width=0.7, direction="in")
        plt.scatter(true[:, 1], predict[:, 1], c=error[:, 1], cmap="plasma")
        plt.plot([0, 0.1], [0, 0.1], linestyle="--", c="white")
        plt.xlabel(r"$\Phi_{\mathrm{GN}}" + "$ true (eV)", size=15)
        plt.ylabel(r"$\Phi_{\mathrm{GN}}" + "$ predicted (eV)", size=15)
        cbar = plt.colorbar()
        cbar.set_label("MAE", rotation=270, labelpad=15, size=15)
        cbar.ax.minorticks_on()
        cbar.ax.tick_params(labelsize=15)
        cbar.ax.tick_params(which="major", length=7)
        cbar.ax.tick_params(which="minor", length=4)
        cbar.ax.tick_params(which="both", width=1.5)
        plt.savefig(f"phign.pdf")

        true[:, 2] = 100 * true[:, 2]
        error[:, 2] = 1000 * error[:, 2]
        predict[:, 2] = 100 * predict[:, 2]
        plt.clf()
        fig, ax = plt.subplots()
        # ax.set_aspect(1)
        plt.tick_params(which="both", width=1.5, direction="in")
        plt.tick_params(which="major", length=7)
        plt.tick_params(which="minor", length=4)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.minorticks_on()
        plt.tick_params(which="both", width=0.7, direction="in")
        plt.scatter(true[:, 2], predict[:, 2], c=error[:, 2], cmap="plasma")
        plt.plot([0, 0.8], [0, 0.8], linestyle="--", c="white")
        plt.xlabel(r"$\epsilon" + "$ true ", size=15)
        plt.ylabel(r"$\epsilon" + "$ predicted ", fontsize="large")
        cbar = plt.colorbar()
        cbar.set_label("MAE", rotation=270, labelpad=15, size=15)
        cbar.ax.minorticks_on()
        cbar.ax.tick_params(labelsize=15)
        cbar.ax.tick_params(which="major", length=7)
        cbar.ax.tick_params(which="minor", length=4)
        cbar.ax.tick_params(which="both", width=1.5)
        plt.savefig(f"epsilon.pdf")

        # plt.clf()
        # plt.minorticks_on()
        # plt.tick_params(which='both',width=0.7,direction='in')
        # plt.scatter(true[:, 3], predict[:, 3], c=error[:, 3], cmap="plasma")
        # plt.plot([0.5, 1.0], [0.5, 1.0])
        # plt.xlabel(r"$\cos\phi" + "$ true", fontsize='large')
        # plt.ylabel(r"$\cos\phi" + "$ predicted", fontsize='large')
        # cbar = plt.colorbar()
        # cbar.set_label("error", rotation=270, labelpad=15)
        # plt.savefig(f"strainpredictioncosphi.pdf")
        ## plt.show()

        # plt.clf()
        # plt.minorticks_on()
        # plt.tick_params(which='both',width=0.7,direction='in')
        # plt.scatter(true[:, 4], predict[:, 4], c=error[:, 4], cmap="plasma")
        # plt.plot([0, np.sqrt(3)/2], [0, np.sqrt(3)/2])
        # plt.xlabel(r"$\sin\phi" + "$ true", fontsize='large')
        # plt.ylabel(r"$\sin\phi" + "$ predicted", fontsize='large')
        # cbar = plt.colorbar()
        # cbar.set_label("error", rotation=270, labelpad=15)
        # plt.savefig(f"strainpredictionsinphi.pdf")

        # Prediction of angle
        true2 = []
        mse2 = []
        error2 = []
        test2 = []
        counter = 0
        for images, labels in self.test_ds2.take(test_size):
            true2.append(
                labels.numpy(),
            )
            cos = predict[counter][3]
            sin = predict[counter][4]
            temp = math.atan2(sin, cos)
            msetemp = np.square(np.subtract(labels.numpy(), temp)).mean() * np.array(
                [1] * len(true2[0])
            )
            ## This isn't physical, it's just the way we use temp = temp%np.pi
            if abs(msetemp) > 1.0:
                print("here", temp)
                temp = -temp

            test2.append(temp)
            mse2.append(
                np.square(np.subtract(labels.numpy(), temp)).mean()
                * np.array([1] * len(true2[0]))
            )
            error2.append(np.abs(np.subtract(labels.numpy(), temp)))
            counter += 1

        true2 = np.array([item for sublist in true2 for item in sublist])
        mse2 = np.array([item for sublist in mse2 for item in sublist])
        error2 = np.array([item for sublist in error2 for item in sublist])

        plt.clf()
        fig, ax = plt.subplots()
        ax.set_aspect(1)
        plt.tick_params(which="both", width=1.5, direction="in")
        plt.tick_params(which="major", length=7)
        plt.tick_params(which="minor", length=4)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.minorticks_on()
        plt.tick_params(which="both", width=0.7, direction="in")
        # plt.scatter(true2, error2, c=true[:, 2], cmap="plasma_r")
        # plt.scatter(true[:, 2], error2, cmap='viridis', s= 34, edgecolors='black', linewidth = 0.5)
        # plt.scatter(true[:, 2], error2, c=error2, cmap='tab20c')
        plt.scatter(true[:, 2], error2, c=error2, cmap="magma")
        # plt.scatter(true[:, 2], error2)
        plt.xlabel(r"$\epsilon" + "$ true", size=15)
        plt.ylabel(r"MAE $\theta_{\varepsilon}$", size=15)
        # cbar = plt.colorbar()
        # cbar.set_label(r"$\epsilon$", rotation=270, labelpad=15)
        # plt.grid()
        plt.tight_layout()
        plt.savefig(f"mae.pdf")

        plt.clf()
        fig, ax = plt.subplots()
        # ax.set_aspect(1)
        plt.tick_params(which="both", width=1.5, direction="in")
        plt.tick_params(which="major", length=7)
        plt.tick_params(which="minor", length=4)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.minorticks_on()
        plt.scatter(true2, test2, c=true[:, 2], cmap="viridis_r")
        plt.plot([0, np.pi / 3], [0, np.pi / 3], linestyle="--", c="white")
        plt.xlabel(r"$\theta_{\varepsilon}" + "$ true", size=15)
        plt.ylabel(r"$\theta_{\varepsilon}" + "$ predicted", size=15)
        cbar = plt.colorbar()
        cbar.set_label(
            r"$\varepsilon\:\left(\%\right)$", rotation=270, labelpad=15, size=15
        )
        cbar.ax.minorticks_on()
        cbar.ax.tick_params(labelsize=15)
        cbar.ax.tick_params(which="major", length=7)
        cbar.ax.tick_params(which="minor", length=4)
        cbar.ax.tick_params(which="both", width=1.5)
        plt.tight_layout()
        plt.savefig(f"theta.pdf")

        plt.clf()
        fig, ax = plt.subplots()
        plt.tick_params(which="major", length=7)
        plt.tick_params(which="minor", length=4)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.minorticks_on()
        plt.minorticks_on()
        plt.tick_params(which="both", width=0.7, direction="in")
        # plt.scatter(true[:, 4], predict[:, 4], c=error[:, 4], cmap="plasma")
        plt.hist(error2)
        plt.xlabel(r"count", size=15)
        plt.ylabel(r"MAE $\theta_{\varepsilon}$", size=15)
        plt.savefig(f"hist.pdf")

        true[:, 2] = 100 * true[:, 2]
        error[:, 2] = 1000 * error[:, 2]
        predict[:, 2] = 100 * predict[:, 2]
        plt.clf()
        fig, ax = plt.subplots()
        # ax.set_aspect(1)
        plt.tick_params(which="both", width=1.5, direction="in")
        plt.tick_params(which="major", length=7)
        plt.tick_params(which="minor", length=4)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.minorticks_on()
        plt.tick_params(which="both", width=0.7, direction="in")
        plt.scatter(true[:, 2], predict[:, 2], c=error2, cmap="plasma")
        plt.plot([0, 0.8], [0, 0.8], linestyle="--", c="white")
        plt.xlabel(r"$\epsilon" + "$ true ", size=15)
        plt.ylabel(r"$\epsilon" + "$ predicted ", fontsize="large")
        cbar = plt.colorbar()
        cbar.set_label("MAE", rotation=270, labelpad=15, size=15)
        cbar.ax.minorticks_on()
        cbar.ax.tick_params(labelsize=15)
        cbar.ax.tick_params(which="major", length=7)
        cbar.ax.tick_params(which="minor", length=4)
        cbar.ax.tick_params(which="both", width=1.5)
        plt.savefig(f"epsilontest.pdf")

        plt.clf()
        # cm = plt.cm.get_cmap('tab20c')
        cm = plt.cm.get_cmap("magma")
        fig, ax = plt.subplots()
        ax.set_aspect(1)
        plt.tick_params(which="major", length=7)
        plt.tick_params(which="minor", length=4)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.minorticks_on()
        plt.minorticks_on()
        plt.tick_params(which="both", width=0.7, direction="in")
        # Get the histogramp
        Y, X = np.histogram(error2, 25)
        plt.ylabel(r"Probability", size=15)
        plt.xlabel(r"MAE $\theta_{\varepsilon}$", size=15)
        x_span = X.max() - X.min()
        Y = Y / test_size
        temp = 0
        for i in range(7):
            temp += Y[i]
        print(temp)
        C = [cm(((x - X.min()) / x_span)) for x in X]
        # plt.text(0.5, 0.7, r'P(0<MAE<0.25)='+str(np.round(100*temp,1))+'%', bbox=dict(facecolor='green', alpha=0.1), transform=plt.gca().transAxes)
        plt.bar(X[:-1], Y, color=C, width=X[1] - X[0])
        plt.xlim(-0.025, 0.5)
        plt.tight_layout()
        plt.savefig(f"hist2.pdf")


def main_train(model: str, batch_size: int, epochs: int) -> None:
    """
    Main function for training the neural network
    """
    ml = ML(model, batch_size, epochs)

    # ml.process_training_data()
    #### ml.plot_training_data()
    # ml.create_model()
    # ml.train_model()
    # ml.plot_history()

    # ml.process_test_data()
    ##ml.evaluate_model()
    ### # # ml.image_error()
    # ml.analyize_model()

    # ml.process_training_data_7channels()
    # ml.create_model_7channels()
    # ml.train_model()
    # ml.process_test_data_7channels()
    ml.process_test_data_5channels()
    # ml.evaluate_model()
    # ml.analyize_model()
    ml.analyize_model_exp()


if __name__ == "__main__":

    cfg = {
        "model": "modelstrain.h5",
        "batch_size": 64,
        "epochs": 2000,
    }

    main_train(
        cfg["model"],
        cfg["batch_size"],
        cfg["epochs"],
    )
