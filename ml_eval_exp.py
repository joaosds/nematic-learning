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
from matplotlib.gridspec import GridSpec
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
from matplotlib import rc
import matplotlib
import math

from matplotlib import rc
import matplotlib.ticker

rc("font", family="serif", serif="cm10")
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]

matplotlib.rcParams["axes.linewidth"] = 2.4  # width of frames

# Plot label font configuration
# rc('font',**{'family':'serif','serif':['Helvetica']})
# rc('text', usetex=True)

# font = {'family' : 'normal',
#        'weight' : 'bold',
#        'size'   : 45}


def norm_and_hist_plots(
    ds: np.ndarray,
    ds_dim: int,
    norm=True,
    px=65,
    plt_hist=True,
    n_channels=7,
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

    means = np.tile(0, n_channels)
    stds = np.tile(1, n_channels)
    # means = np.array([1.76, 1.56, 1.80, 14.52, 66.54, 12.79, 1.61])
    # means = np.array([8.16, 8.61, 9.21, 38.88, 56.14, 41.02, 42.23])

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
            # gs = gridspec(2, 2, figure=fig)
            # ax1 = fig.add_subplot(gs[0, :])
            for k in range(n_channels):
                hists[k, :] = ds[k, i, :, :].flatten()
                meantemp[k] = hists[k, :].mean()
                stdtemp[k] = hists[k, :].std()
            print(len(meantemp))
            # ver como organizar para 3 plots em cima apenas
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

            # textstr = "\n".join((r"$n_{s}=%.2f$" % (meantemp[0],),))
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
                # transform=axes[0, 3].transaxes,
                fontsize=14,
                verticalalignment="top",
                bbox=props,
            )
            plt.show()
            plt.cla()
            plt.clf()
            plt.close()
    return ds


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
    sigma = 0
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

        self.test_ds = None

        self.history = None

        # JA: call init from model class
        super().__init__()

    def process_data(self) -> None:
        """
        Reading in and processing training and validation data from
        npz files on hard disk.
        """

        exp = True
        if exp:
            nlabels = 8
            px = 65
            testds = np.zeros((7, nlabels, px, px))
            with np.load(
                "/home/jass/Downloads/datasetstrain/same padding/expdata/expdata.npz"
            ) as data:
                #     test_examples_1 = data["DataZ"][0 : 3 * nlabels : 3]
                #     test_examples_2 = data["DataZ"][1 : 3 * nlabels + 1 : 3]
                #     test_examples_3 = data["DataZ"][2 : 3 * nlabels + 2 : 3]
                #     test_examples_4 = data["DataX"][0 : 4 * nlabels : 4]
                #     test_examples_5 = data["DataX"][1 : 4 * nlabels + 1 : 4]
                #     test_examples_6 = data["DataX"][2 : 4 * nlabels + 2 : 4]
                #     test_examples_7 = data["DataX"][3 : 4 * nlabels + 3 : 4]
                testds[0, :, :, :] = data["DataZ"][0 : 3 * nlabels : 3]
                testds[1, :, :, :] = data["DataZ"][1 : 3 * nlabels + 1 : 3]
                testds[2, :, :, :] = data["DataZ"][2 : 3 * nlabels + 2 : 3]
                testds[3, :, :, :] = data["DataX"][0 : 4 * nlabels : 4]
                testds[4, :, :, :] = data["DataX"][1 : 4 * nlabels + 1 : 4]
                testds[5, :, :, :] = data["DataX"][2 : 4 * nlabels + 2 : 4]
                testds[6, :, :, :] = data["DataX"][3 : 4 * nlabels + 3 : 4]
                test_labels = data["DataP"].astype(np.float32)[0:nlabels]

        testds[:, :, :, :] = norm_and_hist_plots(
            testds, nlabels, norm=True, plt_hist=False
        )

        test_dataset = tf.data.Dataset.from_tensor_slices(
            (
                (
                    process_images2(testds[0, :, :, :]),
                    process_images2(testds[1, :, :, :]),
                    process_images2(testds[2, :, :, :]),
                    process_images(testds[3, :, :, :]),
                    process_images(testds[4, :, :, :]),
                    process_images(testds[5, :, :, :]),
                    process_images(testds[6, :, :, :]),
                ),
                test_labels,
            )
        )

        test_dataset2 = test_dataset
        test_ds_size = tf.data.experimental.cardinality(test_dataset).numpy()

        print("Test data size of one energy channel:", test_ds_size)
        self.test_ds = test_dataset.batch(batch_size=1, drop_remainder=True)
        print(self.test_ds)
        self.test_ds2 = test_dataset2.batch(batch_size=1, drop_remainder=True)

    def analyize_model_exp_err(self) -> None:
        """
        Analyze accuracy of nematic model.
        """

        test_size = 8
        true = []
        predict = self.model.predict(self.test_ds.take(test_size))
        counter = 0

        label = np.zeros((test_size))

        label[0] = -0.58
        label[1] = -0.45
        label[2] = -0.32
        label[3] = 0.00
        label[4] = 0.34
        label[5] = 0.47
        label[6] = 0.61
        label[7] = 0.67

        print(true, predict)
        for images, labels in self.test_ds.take(test_size):
            true.append(
                labels.numpy(),
            )
            counter += 1

        true = np.array([item for sublist in true for item in sublist])
        print("here")
        print(true)
        print(np.array(predict))
        plt.clf()
        fig, ax = plt.subplots()
        # ax.set_aspect(1)
        plt.tick_params(which="both", width=1.5, direction="in")
        plt.tick_params(which="major", length=7)
        plt.tick_params(which="minor", length=4)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.minorticks_on()
        plt.ylim(0, 0.15)
        plt.tick_params(which="both", width=0.7, direction="in")
        # plt.scatter(true[:], predict[:, 0])
        plt.plot(
            true[:],
            predict[:, 0],
            marker="s",
            linestyle="--",
            color="black",
        )
        plt.xlabel(r"Filling of the CFB $(n_{s})$", size=15)
        plt.ylabel(r"$\Phi_{\mathrm{MN}}" + "$ predicted", size=15)
        plt.tight_layout()
        plt.savefig(f"phimn_exp2_err.pdf")

        mean = np.array(
            [
                [0.0411883, 0.06893247, 0.00191818],
                [0.03754662, 0.0747593, 0.00279409],
                [0.0424214, 0.0704508, 0.00211338],
                [0.04012687, 0.03037238, 0.00283636],
                [0.05592582, 0.05704093, 0.00162906],
                [0.07364587, 0.03364383, 0.00246957],
                [0.07618995, 0.03131388, 0.00116626],
                [0.0803389, 0.02060873, 0.00128506],
            ]
        )
        std = np.array(
            [
                [0.00365687, 0.00485852, 0.00031212],
                [0.00568503, 0.00182443, 0.00031751],
                [0.00549354, 0.00511263, 0.0001049],
                [0.00510609, 0.00195168, 0.00012235],
                [0.00474071, 0.00109793, 0.00024645],
                [0.00588057, 0.0027129, 0.00034623],
                [0.00431731, 0.00216265, 0.0001475],
                [0.0034788, 0.00401435, 0.00040419],
            ]
        )
        plt.clf()
        fig, ax = plt.subplots()
        # ax.set_aspect(1)
        plt.tick_params(which="both", width=2.5, direction="in")
        plt.tick_params(which="major", length=9)
        plt.tick_params(which="minor", length=6)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.minorticks_on()
        plt.ylim(0, 0.1)
        plt.tick_params(which="both", width=0.7, direction="in")
        plt.errorbar(
            true[:],
            mean[:, 0],
            std[:, 0],
            marker="s",
            linestyle="--",
            color="black",
            label=r"$\Phi_{\mathrm{MN}}$ (Moiré)",
        )
        plt.fill_between(
            true[:],
            mean[:, 0] - std[:, 0],
            mean[:, 0] + std[:, 0],
            alpha=0.6,
            color="gray",
        )
        plt.errorbar(
            true[:],
            mean[:, 1],
            std[:, 1],
            marker="d",
            linestyle="--",
            color="red",
            label=r"$\Phi_{\mathrm{GN}}$ (Graphene)",
        )

        plt.fill_between(
            true[:],
            mean[:, 1] - std[:, 1],
            mean[:, 1] + std[:, 1],
            alpha=0.6,
            color="red",
        )
        plt.xlabel(r"Filling of the CFB $(n_{s})$", size=20)
        plt.ylabel(r"Predicted Nematic Intensities (eV)", size=20)
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(f"phign_exp2.pdf")

        predict[:, 2] = 100 * predict[:, 2]
        mean[:, 2] = 100 * mean[:, 2]
        std[:, 2] = 100 * std[:, 2]

        plt.clf()
        fig, ax = plt.subplots()
        # ax.set_aspect(1)
        plt.tick_params(which="both", width=1.8, direction="in")
        plt.tick_params(which="major", length=8)
        plt.tick_params(which="minor", length=5)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.minorticks_on()
        plt.ylim(0, 0.8)
        plt.tick_params(which="both", width=0.7, direction="in")
        plt.errorbar(
            label,
            mean[:, 2],
            std[:, 2],
            marker="d",
            linestyle="--",
            color="orange",
        )
        plt.fill_between(
            true[:],
            mean[:, 2] - std[:, 2],
            mean[:, 2] + std[:, 2],
            alpha=0.5,
            color="orange",
        )
        plt.xlabel(r"Filling of the CFB $(n_{s})$", size=20)
        plt.ylabel(r"Predicted Strain strength $\epsilon$ (\%)", size=20)
        # current_values = plt.gca().get_xticks()
        # plt.gca().set_xticklabels(['{:,.2f}'.format(x) for x in current_values])
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"epsilon_exp2_err.pdf")


    def analyize_model_exp(self) -> None:
        """
        Analyze accuracy of nematic model.
        """

        test_size = 8
        true = []
        predict = self.model.predict(self.test_ds.take(test_size))
        counter = 0

        label = np.zeros((test_size))

        label[0] = -0.58
        label[1] = -0.45
        label[2] = -0.32
        label[3] = 0.00
        label[4] = 0.34
        label[5] = 0.47
        label[6] = 0.61
        label[7] = 0.67

        print(true, predict)
        for images, labels in self.test_ds.take(test_size):
            true.append(
                labels.numpy(),
            )
            counter += 1

        true = np.array([item for sublist in true for item in sublist])
        print("here")
        print(true)
        print(np.array(predict))
        plt.clf()
        fig, ax = plt.subplots()
        # ax.set_aspect(1)
        plt.tick_params(which="both", width=1.5, direction="in")
        plt.tick_params(which="major", length=7)
        plt.tick_params(which="minor", length=4)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.minorticks_on()
        plt.ylim(0, 0.15)
        plt.tick_params(which="both", width=0.7, direction="in")
        # plt.scatter(true[:], predict[:, 0])
        plt.plot(
            true[:],
            predict[:, 0],
            marker="s",
            linestyle="--",
            color="black",
        )
        plt.xlabel(r"Filling of the CFB $(n_{s})$", size=15)
        plt.ylabel(r"$\Phi_{\mathrm{MN}}" + "$ predicted", size=15)
        plt.tight_layout()
        plt.savefig(f"phimn_exp2.pdf")

        # plt.clf()
        # fig, ax = plt.subplots()
        # # ax.set_aspect(1)
        # plt.tick_params(which="both", width=1.5, direction="in")
        # plt.tick_params(which="major", length=7)
        # plt.tick_params(which="minor", length=4)
        # plt.xticks(fontsize=15)
        # plt.yticks(fontsize=15)
        # plt.minorticks_on()
        # # plt.ylim(0, 0.08)
        # plt.tick_params(which="both", width=0.7, direction="in")
        # plt.plot(
        #     true[:],
        #     predict[:, 1],
        #     marker="s",
        #     linestyle="--",
        #     color="cyan",
        #     label=r"$\sin_{\varphi_{\mathrm{MN}}}$",
        # )
        # plt.plot(
        #     true[:],
        #     predict[:, 2],
        #     marker="d",
        #     linestyle="--",
        #     color="green",
        #     label=r"$\cos{\varphi_{\mathrm{MN}}}$",
        # )
        # plt.xlabel(r"Filling of the CFB $(n_{s})$", size=15)
        # # plt.ylabel(r"$\Phi_{\mathrm{GN}}" + "$ predicted (eV)", size=15)
        # plt.ylabel(r"Predicted sin/cos of $\varphi$", size=15)
        # plt.legend(1loc="lower left")
        # plt.tight_layout()
        # plt.savefig(f"phisincos_2.pdf")

        mean = np.array(
            [
                [0.0411883, 0.06893247, 0.00191818],
                [0.03754662, 0.0747593, 0.00279409],
                [0.0424214, 0.0704508, 0.00211338],
                [0.04012687, 0.03037238, 0.00283636],
                [0.05592582, 0.05704093, 0.00162906],
                [0.07364587, 0.03364383, 0.00246957],
                [0.0803389, 0.02060873, 0.00128506],
            ]
        )
        std = np.array(
            [
                [0.00365687, 0.00485852, 0.00031212],
                [0.00568503, 0.00182443, 0.00031751],
                [0.00549354, 0.00511263, 0.0001049],
                [0.00510609, 0.00195168, 0.00012235],
                [0.00474071, 0.00109793, 0.00024645],
                [0.00588057, 0.0027129, 0.00034623],
                [0.0034788, 0.00401435, 0.00040419],
            ]
        )
        plt.clf()
        fig, ax = plt.subplots()
        # ax.set_aspect(1)
        plt.tick_params(which="both", width=2.5, direction="in")
        plt.tick_params(which="major", length=8)
        plt.tick_params(which="minor", length=5)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.minorticks_on()
        plt.ylim(0, 0.1)
        plt.tick_params(which="both", width=0.7, direction="in")
        plt.plot(
            true[:],
            predict[:, 0],
            marker="s",
            linestyle="--",
            color="black",
            label=r"$\Phi_{\mathrm{MN}}$ (Moiré)",
        )
        # plt.errorbar(
        #     true[:],
        #     mean[:, 0],
        #     std[:, 0],
        #     marker="s",
        #     linestyle="--",
        #     color="black",
        #     label=r"$\Phi_{\mathrm{MN}}$ (Moiré)",
        # )
        # plt.fill_between(
        #     true[:],
        #     mean[:, 0] - std[:, 0],
        #     mean[:, 0] + std[:, 0],
        #     alpha=0.6,
        #     color="gray",
        # )
        # plt.errorbar(
        #     true[:],
        #     mean[:, 1],
        #     std[:, 1],
        #     marker="d",
        #     linestyle="--",
        #     color="red",
        #     label=r"$\Phi_{\mathrm{GN}}$ (Graphene)",
        # )

        # plt.fill_between(
        #     true[:],
        #     mean[:, 1] - std[:, 1],
        #     mean[:, 1] + std[:, 1],
        #     alpha=0.6,
        #     color="red",
        # )
        plt.plot(
            true[:],
            predict[:, 1],
            marker="d",
            linestyle="--",
            # color="black",
            # label=r"$\Phi_{\mathrm{MN}}$ (Moiré)",
            color="red",
            label=r"$\Phi_{\mathrm{GN}}$ (Graphene)",
        )
        plt.xlabel(r"Filling of the CFB $(n_{s})$", size=18)
        # plt.ylabel(r"$\Phi_{\mathrm{GN}}" + "$ predicted (eV)", size=15)
        plt.ylabel(r"Predicted Nematic Intensities (eV)", size=18)
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(f"phign_exp2.pdf")

        predict[:, 2] = 100 * predict[:, 2]
        mean[:, 2] = 100 * mean[:, 2]
        std[:, 2] = 100 * std[:, 2]

        plt.clf()
        fig, ax = plt.subplots()
        # ax.set_aspect(1)
        plt.tick_params(which="both", width=1.8, direction="in")
        plt.tick_params(which="major", length=8)
        plt.tick_params(which="minor", length=5)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.minorticks_on()
        plt.ylim(0, 0.8)
        plt.tick_params(which="both", width=0.7, direction="in")
        plt.plot(
            true[:],
            predict[:, 2],
            marker="d",
            linestyle="--",
            # color="black",
            # label=r"$\Phi_{\mathrm{MN}}$ (Moiré)",
            color="orange",
        )
        # plt.errorbar(
        #     label,
        #     mean[:, 2],
        #     std[:, 2],
        #     marker="d",
        #     linestyle="--",
        #     color="orange",
        # )
        # plt.fill_between(
        #     true[:],
        #     mean[:, 2] - std[:, 2],
        #     mean[:, 2] + std[:, 2],
        #     alpha=0.5,
        #     color="yellow",
        # )
        # plt.plot([-0.58, 0.67], [0, 0.8], linestyle="--", c="white")
        plt.xlabel(r"Filling of the CFB $(n_{s})$", size=18)
        plt.ylabel(r"Predicted Strain strength $\epsilon$ (\%)", size=18)
        # current_values = plt.gca().get_xticks()
        # plt.gca().set_xticklabels(['{:,.2f}'.format(x) for x in current_values])
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"epsilon_exp2.pdf")

        plt.clf()
        fig, ax = plt.subplots()
        # ax.set_aspect(1)
        plt.tick_params(which="both", width=1.8, direction="in")
        plt.tick_params(which="major", length=10)
        plt.tick_params(which="minor", length=6)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.minorticks_on()
        plt.ylim(0, 1)
        plt.tick_params(which="both", width=0.7, direction="in")
        plt.scatter(true[:], predict[:, 3])
        # plt.plot([-0.58, 0.67], [0, 0.8], linestyle="--", c="white")
        plt.xlabel(r"Filling of the CFB $(n_{s})$", size=20)
        plt.ylabel(r"$\theta_{epsilon}" + "$ predicted ", size=20)
        plt.tight_layout()
        plt.savefig(f"theta_epsilon_exp2.pdf")
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
        # plt.yrange(0, 1)
        plt.plot(
            true2,
            test2,
            marker="d",
            linestyle="--",
            color="pink",
        )
        plt.minorticks_on()
        plt.scatter(true2, test2)
        plt.xlabel(r"Filling of the CFB $(n_{s})$", size=15)
        plt.ylabel(r"$\theta_{\varepsilon}" + "$ predicted", size=15)
        plt.tight_layout()
        plt.savefig(f"theta_exp2.pdf")


def main_train(model: str, batch_size: int, epochs: int) -> None:
    """
    Main function for training the neural network
    """
    ml = ML(model, batch_size, epochs)

    ml.process_data()
    # ml.analyize_model_exp()
    ml.analyize_model_exp_err()


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
