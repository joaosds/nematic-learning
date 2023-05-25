#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from matplotlib import rc
import matplotlib
import matplotlib.ticker
import math
import scipy

rc("font", family="serif", serif="cm10")
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
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
    mape = np.around(np.mean(np.abs((np.abs(y_test) - np.abs(pred)) / y_test)), 3)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_test, pred)
    rsq = np.around(r_value**2, 3)
    print(rsq, mape)
    return mape, rsq


def norm_and_hist_plots(
    ds: np.ndarray,
    ds_dim: int,
    norm=True,
    px=65,
    plt_hist=True,
    n_channels=7,
    disorder=False,
    n_dis=3,
) -> np.ndarray:
    """
    function to normalize distributions to be centered at 0 (mean) with a standard deviation of 1 if norm = 1. if plt_hist=true,
    the histogram plot of pixel intensities is plotted for each energy channel

    parameters
    ----------
    ds: np.ndarray
        dataset of dimensions (n_channels, ds_dim, px, px) containing images that will be normalized.
    means : np.ndarray
        list of desired means for each channel.
    stds : np.ndarray
        list of desired standard deviations for each channel.
    px : integer
        pixel size of dos(r) images
    plt_hist : boolean
        option to plot histograms (true) or not (false).
    n_channels : int
        number of channels in the dataset sd.
    norm: boolean
        normalize (1) or not the images.

    returns
    -------
    ds : np.ndarray
        normalized dataset
    """

    def gaussian(i: float, x: float, y: float, sigma: float = 2) -> float:
        """
        Computes the intensity of disorder using a Gaussian function.

        Parameters:
            i (float): The intensity of the disorder.
            x (float): The x-coordinate.
            y (float): The y-coordinate.
            sigma (float, optional): The standard deviation for the Gaussian. Default is 2.

        Returns:
            float: The computed intensity of strain.

        Notes:
            The Gaussian function is given by:
            a = exp(-(r / (2 * sigma^2))) / sqrt((2 * pi * sigma^2))
            where r = x^2 + y^2.

        Example:
            >>> i = 1.5
            >>> x = 2.0
            >>> y = 3.0
            >>> sigma = 2
            >>> result = gaussian(i, x, y, sigma)
            >>> print(result)
            0.0471683980329
        """
        r = x**2 + y**2
        a = np.exp(-(r / (2 * sigma**2))) / np.sqrt((2 * np.pi * sigma**2))
        return a * i

    means = np.tile(0, n_channels)
    stds = np.tile(1, n_channels)

    radius = 5
    if disorder:
        for j in range(ds_dim):
            x_dis = np.array(random.sample(range(5, 60), n_dis))
            y_dis = np.array(random.sample(range(5, 60), n_dis))
            for i in range(n_dis):
                for k1 in range(-radius, radius + 1):
                    for k2 in range(-radius, radius + 1):
                        ds[:, j, x_dis[i] + k2, y_dis[i] + k1] = ds[
                            :, j, x_dis[i] + k2, y_dis[i] + k1
                        ] + gaussian(100, k1, k2, 3)

    if norm:
        for k in range(n_channels):
            for i in range(ds_dim):
                ds[k, i, :, :] = (
                    means[k]
                    + (ds[k, i, :, :] - ds[k, i, :, :].flatten().mean())
                    * stds[k]
                    / ds[k, i, :, :].flatten().std()
                )

    if plt_hist:
        meantemp = np.zeros((n_channels))
        stdtemp = np.zeros((n_channels))
        hists = np.zeros((n_channels, px * px))
        for i in range(ds_dim):
            _, axes = plt.subplots(2, 4)
            for k in range(n_channels):
                hists[k, :] = ds[k, i, :, :].flatten()
                meantemp[k] = hists[k, :].mean()
                stdtemp[k] = hists[k, :].std()
            print(len(meantemp))
            axes[0, 0].set_title(r"baac")
            axes[0, 1].set_title(r"abca")
            axes[0, 2].set_title(r"abab")
            axes[0, 3].axis("off")
            axes[1, 0].set_title(r"e = -35 mev (rv$_{1}$)")
            axes[1, 1].set_title(r"e = -15 mev (vfb)")
            axes[1, 2].set_title(r"e = 1 mev (cfb)")
            axes[1, 3].set_title(r"e = 23 mev (rc$_{1}$)")
            axes[0, 0].hist(hists[0, :], bins=50, facecolor="purple")
            axes[0, 1].hist(hists[1, :], bins=50, facecolor="black")
            axes[0, 2].hist(hists[2, :], bins=50, facecolor="red")
            axes[1, 0].hist(hists[3, :], bins=50)
            axes[1, 1].hist(hists[4, :], bins=50)
            axes[1, 2].hist(hists[5, :], bins=50)
            axes[1, 3].hist(hists[6, :], bins=50)

            textstr = "\n".join(
                (
                    r"\textbf{before norm}",
                    "",
                    "scaleograms:",
                    "",
                    r"$\mu=%.2f, %.2f, %.2f$" % (meantemp[0], meantemp[1], meantemp[2]),
                    r"$\sigma=%.2f, %.2f, %.2f$" % (stdtemp[0], stdtemp[1], stdtemp[2]),
                    r"$\mu_{t}=%.2f, \sigma_{s}=%.2f$"
                    % (meantemp[0:2].mean(), stdtemp[0:2].std()),
                    "",
                    "dos(r):",
                    "",
                    r"$\mu=%.2f, %.2f, %.2f, %.2f$"
                    % (meantemp[3], meantemp[4], meantemp[5], meantemp[6]),
                    r"$\sigma=%.2f, %.2f, %.2f, %.2f$"
                    % (stdtemp[3], stdtemp[4], stdtemp[5], stdtemp[6]),
                    r"$\mu_{t}=%.2f, \sigma_{s}=%.2f$"
                    % (meantemp[3:6].mean(), stdtemp[3:6].std()),
                    "",
                    "full dataset",
                    "",
                    r"$\mu_{ds}=%.2f, \sigma_{ds}=%.2f$"
                    % (meantemp.mean(), stdtemp.std()),
                )
            )
            # these are matplotlib.patch.patch properties
            props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

            # place a text box in upper left in axes coords
            axes[0, 3].text(
                0.05,
                0.95,
                textstr,
                transform=axes[0, 3].transaxes,
                fontsize=14,
                verticalalignment="top",
                bbox=props,
            )
            plt.show()
            plt.cla()
            plt.clf()
            plt.close()
    return ds


def process_images2(image):
    """
    Add additional channel axis to image. Convert image from float64 to float32.
    Add gaussian noise. (For dos_{r}(\omega) we added noise in the TDBG.py file)
    """
    image = image[..., tf.newaxis]  # [height, width, channels]
    image = tf.image.convert_image_dtype(image, tf.float32)
    batch, row, col, ch = image.shape
    mean = 0
    sigma = 0
    gauss = np.abs(np.random.normal(mean, sigma, (batch, row, col, ch)))
    gauss = gauss.reshape(batch, row, col, ch)
    image = image + gauss
    return image


def process_images(image):
    """
    Add additional channel axis to image. Convert image from float64 to float32.
    Add gaussian noise.
    """
    image = image[..., tf.newaxis]  # [height, width, channels]
    image = tf.image.convert_image_dtype(image, tf.float32)
    batch, row, col, ch = image.shape
    mean = 0
    sigma = 0.31
    gauss = np.abs(np.random.normal(mean, sigma, (batch, row, col, ch)))
    gauss = gauss.reshape(batch, row, col, ch)
    for i, j in range(row, col):
        print(i, j)
        gauss[:, i, j, :] = 0
    image = image + gauss
    return image


def plot_history() -> None:
    """
    The fit() method returns a History object containing the training
    parameters (history.params), the list of epochs it went through
    (history.epoch), and most importantly a dictionary (history.history)
    containing the loss and extra metrics it measured at the end of each
    epoch on the training set and on the validation set.
    """
    with open("history.pkl", "rb") as file:
        history = pickle.load(file)
    loss, acc, val_loss, val_acc = list(history.values())
    start = 1  # at which epoch the plot should start
    plt.clf()
    plt.minorticks_on()
    plt.tick_params(which="both", width=0.7, direction="in")
    plt.plot(range(start, len(loss)), loss[start:], "r", label="Training")
    plt.plot(range(start, len(val_loss)), val_loss[start:], "b", label="Validation")
    plt.legend()
    plt.xlabel(r"Epochs", fontsize="large")
    plt.ylabel(r"Loss", fontsize="large")
    plt.savefig("loss.pdf")

    plt.clf()
    plt.minorticks_on()
    plt.tick_params(which="both", width=0.7, direction="in")
    plt.plot(range(1, len(acc) + 1), acc, "r", label="Train")
    plt.plot(range(1, len(val_acc) + 1), val_acc, "b", label="Validation")
    plt.legend(loc="upper right", prop={"size": 11}, markerscale=5)
    plt.legend()
    plt.xlabel(r"Epochs", fontsize="large")
    plt.ylabel(r"MAE", fontsize="large")
    plt.savefig("strainmae.pdf")


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

        super().__init__()

    def plot_training_data(self) -> None:
        """
        Plotting examples of the training data.
        """
        cmap = plt.cm.inferno.copy()
        for images, labels in self.train_ds.take(1):
            for i in range(0, 10):
                plt.clf()
                plt.title(np.round(labels[i].numpy(), 4), fontsize=12)
                plt.imshow(images[0][i].numpy(), cmap=cmap)
                plt.colorbar()
                plt.axis("off")
                plt.savefig(f"channel1_{i}.pdf")
                plt.show()
                plt.clf()
                plt.title(np.round(labels[i].numpy(), 4), fontsize=12)
                plt.imshow(images[1][i].numpy(), cmap=cmap)
                plt.colorbar()
                plt.axis("off")
                plt.savefig(f"channel2_{i}.pdf")
                plt.clf()
                plt.imshow(images[2][i].numpy(), cmap=cmap)
                plt.colorbar()
                plt.axis("off")
                plt.savefig(f"channel3_{i}.pdf")
                plt.clf()
                plt.imshow(images[3][i].numpy(), cmap=cmap)
                plt.colorbar()
                plt.axis("off")
                plt.savefig(f"channel4_{i}.pdf")

    def create_model_7channels(self) -> None:
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
        # fith branch
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
        # sitxh branch
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
        # apply a FC layer and then a regression prediction on the
        # combined outputs
        z = keras.layers.Dense(4096, activation="relu")(combined)
        z = keras.layers.Dropout(0.2)(z)
        z = keras.layers.Dense(5, activation="linear")(z)
        # our model will accept the inputs of the branches and
        # then output a single value
        self.model = keras.Model(
            inputs=[x.input, y.input, a.input, b.input, i.input, p.input, q.input],
            outputs=z,
        )
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
            patience=60, restore_best_weights=True
        )

        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.epochs,
            callbacks=[
                checkpoint_cb,
                early_stopping_cb,
            ],
        )

        # save training history
        with open("history.pkl", "wb") as file:
            pickle.dump(self.history.history, file)

    def evaluate_model(self) -> None:
        """
        Evaluate model on the test set to estimate the generalization error
        """
        keras.utils.plot_model(self.model, to_file="model3points.png")
        self.model.evaluate(self.test_ds)
        test = self.test_ds.take(1)
        for images, labels in test:
            for i in range(1):
                plt.title(
                    "True: "
                    + str(np.round(labels[i].numpy(), 3))
                    + "\nPred.: "
                    + str(np.round(self.model.predict(test)[i], 3)),
                    fontsize=12,
                )
                plt.imshow(images[0][i].numpy(), cmap="inferno")
                plt.colorbar()
                plt.axis("off")
                plt.savefig("example.pdf")

    def analyze_model(self) -> None:
        """
        Analyze accuracy of nematic model.
        """

        test_size = 2000
        true = []
        predict = self.model.predict(self.test_ds.take(test_size))
        mse = []
        error = []
        eps = []
        counter = 0

        for _, labels in self.test_ds.take(test_size):
            true.append(
                labels.numpy(),
            )
            mse.append(
                np.square(np.subtract(labels.numpy(), predict[counter])).mean()
                * np.array([1] * len(true[0]))
            )
            eps.append(predict[counter][2])
            error.append(np.abs(np.subtract(labels.numpy(), predict[counter])))
            counter += 1

        true = np.array([item for sublist in true for item in sublist])
        mse = np.array([item for sublist in mse for item in sublist])
        error = np.array([item for sublist in error for item in sublist])

        font_size = 16  # Adjust as appropriate.
        font_sizemae = 14  # Adjust as appropriate.

        plt.clf()
        plt.tick_params(which="both", width=2.5, direction="in")
        plt.tick_params(which="major", length=7)
        plt.tick_params(which="minor", length=4)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.minorticks_on()
        plt.minorticks_on()
        plt.scatter(true[:, 0], predict[:, 0], c=error[:, 0], cmap="plasma")
        plt.plot([0, 0.1], [0, 0.1], "w--")
        plt.xlabel(r"$\Phi_{\text{MN}}" + "$ true (eV)", fontsize=font_size)
        plt.ylabel(r"$\Phi_{\text{MN}}" + "$ predicted (eV)", fontsize=font_size)
        cbar = plt.colorbar(format=OOMFormatter(-3, mathText=True))
        cbar.set_label("MAE", rotation=270, labelpad=20, fontsize=font_sizemae)
        cbar.ax.tick_params(labelsize=font_sizemae)
        mape, rsq = stat(true[:, 0], predict[:, 0])
        metrics = f"R$^{2}$: " + str(rsq) + "\n" + "MAPE: " + str(mape)
        # For position with respect to axis and not labels
        x0, xmax = plt.xlim()
        y0, _ = plt.ylim()
        data_width = np.abs(xmax) - np.abs(x0)
        plt.text(
            np.abs(data_width) - 2 * np.abs(x0),
            np.abs(y0),
            metrics,
            bbox=dict(facecolor="none", edgecolor="black", boxstyle="round,pad=1"),
            fontsize=font_sizemae,
            ha="right",
        )
        plt.tight_layout()
        plt.savefig(f"strainPredictionPhiMN.pdf")

        plt.clf()
        plt.tick_params(which="both", width=2.5, direction="in")
        plt.tick_params(which="major", length=7)
        plt.tick_params(which="minor", length=4)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.minorticks_on()
        plt.minorticks_on()
        plt.scatter(true[:, 1], predict[:, 1], c=error[:, 1], cmap="plasma")
        plt.plot([0, 0.1], [0, 0.1], "w--")
        plt.xlabel(r"$\Phi_{\text{GN}}" + "$ true (eV)", fontsize=font_size)
        plt.ylabel(r"$\Phi_{\text{GN}}" + "$ predicted (eV)", fontsize=font_size)
        cbar = plt.colorbar(format=OOMFormatter(-3, mathText=True))
        cbar.set_label("MAE", rotation=270, labelpad=20, fontsize=font_sizemae)
        cbar.ax.tick_params(labelsize=font_sizemae)
        mape, rsq = stat(true[:, 1], predict[:, 1])
        metrics = f"R$^{2}$: " + str(rsq) + "\n" + "MAPE: " + str(mape)
        # For position with respect to axis and not labels
        x0, xmax = plt.xlim()
        y0, _ = plt.ylim()
        data_width = np.abs(xmax) - np.abs(x0)
        plt.text(
            np.abs(data_width) - 2 * np.abs(x0),
            np.abs(y0),
            metrics,
            bbox=dict(facecolor="none", edgecolor="black", boxstyle="round,pad=1"),
            fontsize=font_sizemae,
            ha="right",
        )
        plt.tight_layout()
        plt.savefig(f"strainPredictionPhiIAGN.pdf")

        plt.clf()
        plt.tick_params(which="both", width=2.5, direction="in")
        plt.tick_params(which="major", length=7)
        plt.tick_params(which="minor", length=4)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.minorticks_on()
        plt.minorticks_on()
        plt.scatter(true[:, 2], predict[:, 2], c=error[:, 2], cmap="plasma")
        plt.ticklabel_format(axis="both", style="sci", scilimits=(-3, -3))
        plt.plot([0, 0.008], [0, 0.008], "w--")
        plt.xlabel(r"$\epsilon" + "$ true", fontsize=font_size)
        plt.ylabel(r"$\epsilon" + "$ predicted", fontsize=font_size)
        cbar = plt.colorbar(format=OOMFormatter(-3, mathText=True))
        cbar.set_label("MAE", rotation=270, labelpad=20, fontsize=font_sizemae)
        cbar.ax.tick_params(labelsize=font_sizemae)
        mape, rsq = stat(np.abs(predict[:, 2]), np.abs(true[:, 2]))
        metrics = f"R$^{2}$: " + str(rsq) + "\n" + "MAPE: " + str(mape)
        # For position with respect to axis and not labels
        x0, xmax = plt.xlim()
        y0, _ = plt.ylim()
        data_width = np.abs(xmax) - np.abs(x0)
        plt.text(
            np.abs(data_width) - 2 * np.abs(x0),
            np.abs(y0),
            metrics,
            bbox=dict(facecolor="none", edgecolor="black", boxstyle="round,pad=1"),
            fontsize=font_sizemae,
            ha="right",
        )
        plt.tight_layout()
        plt.savefig(f"strainpredictionepsilon.pdf")

        plt.clf()
        plt.minorticks_on()
        plt.tick_params(which="both", width=0.7, direction="in")
        plt.scatter(true[:, 3], predict[:, 3], c=true[:, 2], cmap="plasma_r")
        plt.plot([0.5, 1.0], [0.5, 1.0])
        plt.xlabel(r"$\cos\phi" + "$ true", fontsize="large")
        plt.ylabel(r"$\cos\phi" + "$ predicted", fontsize="large")
        cbar = plt.colorbar()
        cbar.set_label("error", rotation=270, labelpad=15)
        plt.savefig(f"strainpredictioncosphi.pdf")

        plt.clf()
        plt.minorticks_on()
        plt.tick_params(which="both", width=0.7, direction="in")
        plt.scatter(true[:, 4], predict[:, 4], c=true[:, 2], cmap="plasma_r")
        plt.plot([0, np.sqrt(3) / 2], [0, np.sqrt(3) / 2])
        plt.xlabel(r"$\sin\phi" + "$ true", fontsize="large")
        plt.ylabel(r"$\sin\phi" + "$ predicted", fontsize="large")
        cbar = plt.colorbar()
        cbar.set_label("error", rotation=270, labelpad=15)
        plt.savefig(f"strainpredictionsinphi.pdf")

        # Prediction of angle
        true2 = []
        mse2 = []
        eps2 = []
        error2 = []
        test2 = []
        counter = 0
        for _, labels in self.test_ds2.take(test_size):
            true2.append(labels.numpy())
            eps = predict[counter][2]
            cos = predict[counter][3]
            sin = predict[counter][4]
            temp = math.atan2(sin, cos)
            msetemp = np.square(np.subtract(labels.numpy(), temp)).mean() * np.array(
                [1] * len(true2[0])
            )
            # This isn't physical, it's just the way we use temp = temp%np.pi
            if abs(msetemp) > 1.0:
                print("here", temp)
                temp = -temp

            test2.append(temp)
            mse2.append(
                np.square(np.subtract(labels.numpy(), temp)).mean()
                * np.array([1] * len(true2[0]))
            )
            eps2.append(eps)
            error2.append(np.abs(np.subtract(labels.numpy(), temp)))
            counter += 1

        true2 = np.array([item for sublist in true2 for item in sublist])
        mse2 = np.array([item for sublist in mse2 for item in sublist])
        error2 = np.array([item for sublist in error2 for item in sublist])

        plt.clf()
        plt.tick_params(which="both", width=2.5, direction="in")
        plt.tick_params(which="major", length=7)
        plt.tick_params(which="minor", length=4)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.minorticks_on()
        plt.minorticks_on()
        plt.scatter(true2[:, 0], test2, c=true[:, 2], cmap="viridis_r")
        plt.plot([0, np.pi / 3], [0, np.pi / 3], "w--")
        plt.xlabel(r"$\theta_{\epsilon}" + "$ true", fontsize=font_size)
        plt.ylabel(r"$\theta_{\epsilon}" + "$ predicted", fontsize=font_size)
        cbar = plt.colorbar(format=OOMFormatter(-3, mathText=True))
        cbar.set_label(r"$\epsilon$", rotation=270, labelpad=20, fontsize=font_sizemae)
        cbar.ax.tick_params(labelsize=font_sizemae)
        mape, rsq = stat(test2, true2[:, 0])
        metrics = f"R$^{2}$: " + str(rsq) + "\n" + "MAPE: " + str(mape)
        # For position with respect to axis and not labels
        x0, xmax = plt.xlim()
        y0, _ = plt.ylim()
        data_width = np.abs(xmax) - np.abs(x0)
        plt.text(
            np.abs(data_width) - 2 * np.abs(x0),
            np.abs(y0),
            metrics,
            bbox=dict(facecolor="none", edgecolor="black", boxstyle="round,pad=1"),
            fontsize=font_sizemae,
            ha="right",
        )
        plt.tight_layout()
        plt.savefig(f"strainpredictionphi.pdf")

        plt.clf()
        plt.minorticks_on()
        plt.tick_params(which="both", width=0.7, direction="in")
        plt.scatter(true[:, 2], error2, c=error2, cmap="inferno")
        plt.xlabel(r"count", fontsize="large")
        plt.ylabel(r"MAE", fontsize="large")
        plt.savefig(f"hist23.pdf")
        plt.clf()

    def process_training_data_7channels(self):
        nlabels = 8000
        px = 65
        trainds = np.zeros((7, nlabels, px, px))
        with np.load("sec2e_tra.npz") as data:
            trainds[0, :, :, :] = data["DataW"][0 : 3 * nlabels : 3]
            trainds[1, :, :, :] = data["DataW"][1 : 3 * nlabels + 1 : 3]
            trainds[2, :, :, :] = data["DataW"][2 : 3 * nlabels + 2 : 3]
            trainds[3, :, :, :] = data["DataX"][3 : 4 * nlabels : 4]
            trainds[4, :, :, :] = data["DataX"][4 : 4 * nlabels + 1 : 4]
            trainds[5, :, :, :] = data["DataX"][5 : 4 * nlabels + 2 : 4]
            trainds[6, :, :, :] = data["DataX"][6 : 4 * nlabels + 3 : 4]
            train_labels = data["DataY"].astype(np.float32)[0:nlabels]

        trainds[:, :, :, :] = norm_and_hist_plots(
            ds=trainds, ds_dim=nlabels, norm=True, plt_hist=False
        )

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (
                (
                    process_images2(trainds[0, :, :, :]),
                    process_images2(trainds[1, :, :, :]),
                    process_images2(trainds[2, :, :, :]),
                    process_images2(trainds[3, :, :, :]),
                    process_images2(trainds[4, :, :, :]),
                    process_images2(trainds[5, :, :, :]),
                    process_images2(trainds[6, :, :, :]),
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

        # Creating validation dataset
        valds = np.zeros((7, nlabels, px, px))
        with np.load("sec2e_val.npz") as data:
            valds[0, :, :, :] = data["DataW"][0 : 3 * nlabels : 3]
            valds[1, :, :, :] = data["DataW"][1 : 3 * nlabels + 1 : 3]
            valds[2, :, :, :] = data["DataW"][2 : 3 * nlabels + 2 : 3]
            valds[3, :, :, :] = data["DataX"][3 : 4 * nlabels : 4]
            valds[4, :, :, :] = data["DataX"][4 : 4 * nlabels + 1 : 4]
            valds[5, :, :, :] = data["DataX"][5 : 4 * nlabels + 2 : 4]
            valds[6, :, :, :] = data["DataX"][6 : 4 * nlabels + 3 : 4]
            val_labels = data["DataY"].astype(np.float32)[0:nlabels]

        valds[:, :, :, :] = norm_and_hist_plots(
            ds=valds, ds_dim=nlabels, norm=True, plt_hist=False
        )

        val_dataset = tf.data.Dataset.from_tensor_slices(
            (
                (
                    process_images2(valds[0, :, :, :]),
                    process_images2(valds[1, :, :, :]),
                    process_images2(valds[2, :, :, :]),
                    process_images(valds[3, :, :, :]),
                    process_images(valds[4, :, :, :]),
                    process_images(valds[5, :, :, :]),
                    process_images(valds[6, :, :, :]),
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

    def process_test_data_7channels(self) -> None:
        """
        Reading in and processing training and validation data from
        npz files on hard disk.
        """
        nlabels = 2000
        px = 65
        tesds = np.zeros((7, nlabels, px, px))
        with np.load("sec2e_test.npz") as data:
            tesds[0, :, :, :] = data["DataW"][0 : 3 * nlabels : 3]
            tesds[1, :, :, :] = data["DataW"][1 : 3 * nlabels + 1 : 3]
            tesds[2, :, :, :] = data["DataW"][2 : 3 * nlabels + 2 : 3]
            tesds[3, :, :, :] = data["DataX"][3 : 4 * nlabels : 4]
            tesds[4, :, :, :] = data["DataX"][4 : 4 * nlabels + 1 : 4]
            tesds[5, :, :, :] = data["DataX"][5 : 4 * nlabels + 2 : 4]
            tesds[6, :, :, :] = data["DataX"][6 : 4 * nlabels + 3 : 4]
            tes_labels = data["DataY"].astype(np.float32)[0:nlabels]
            tes_labels2 = data["DataP"].astype(np.float32)[0:nlabels]

        tesds[:, :, :, :] = norm_and_hist_plots(
            ds=tesds, ds_dim=nlabels, norm=True, plt_hist=False
        )
        print(tes_labels.shape)

        test_dataset = tf.data.Dataset.from_tensor_slices(
            (
                (
                    process_images2(tesds[0, :, :, :]),
                    process_images2(tesds[1, :, :, :]),
                    process_images2(tesds[2, :, :, :]),
                    process_images(tesds[3, :, :, :]),
                    process_images(tesds[4, :, :, :]),
                    process_images(tesds[5, :, :, :]),
                    process_images(tesds[6, :, :, :]),
                ),
                tes_labels,
            )
        )

        test_dataset2 = tf.data.Dataset.from_tensor_slices(
            (
                (
                    process_images2(tesds[0, :, :, :]),
                    process_images2(tesds[1, :, :, :]),
                    process_images2(tesds[2, :, :, :]),
                    process_images(tesds[3, :, :, :]),
                    process_images(tesds[4, :, :, :]),
                    process_images(tesds[5, :, :, :]),
                    process_images(tesds[6, :, :, :]),
                ),
                tes_labels2,
            )
        )

        test_ds_size = tf.data.experimental.cardinality(test_dataset).numpy()
        print("Test data size of one energy channel:", test_ds_size)
        self.test_ds = test_dataset.batch(batch_size=1, drop_remainder=True)
        print(self.test_ds)
        self.test_ds2 = test_dataset2.batch(batch_size=1, drop_remainder=True)


def main_train(model: str, batch_size: int, epochs: int) -> None:
    """
    Main function for training the neural network
    """
    ml = ML(model, batch_size, epochs)

    ml.process_training_data_7channels()
    # ml.plot_training_data()
    ml.create_model_7channels()
    ml.train_model()
    ml.process_test_data_7channels()
    ml.analyze_model()


if __name__ == "__main__":
    cfg = {
        "model": "sec2e_modelstrain.h5",
        "batch_size": 64,
        "epochs": 2000,
    }

    main_train(
        cfg["model"],
        cfg["batch_size"],
        cfg["epochs"],
    )
