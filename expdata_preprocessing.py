import numpy as np
import matplotlib.pyplot as plt
from scipy import io
import numpy as np
import sys
import cv2
import matplotlib
import skimage.transform as st
import pywt

import skimage.filters
from matplotlib import rc
import matplotlib.ticker

rc("font", family="serif", serif="cm10")
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]
matplotlib.rcParams["axes.linewidth"] = 2.4  # width of frames

# Reference
# [1] Microscopic nematicity: Rubio-Verdú, C., Turkel, S., Song, Y. et al. Nat. Phys. 18, 196–202 (2022). "Moiré nematic phase in twisted double bilayer graphene.

# For scaleograms
plot_it_full = False
plot_it_partial = False

# To plot position of sites ABCA, BAAC and ABAB in the DOS(r) image
plot_it_post = False

# Plotting all the DOS(r) images
plot_final = False


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
        pixel size of dos(r) images plt_hist : boolean
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


def wavelet_trafo(ldos: list, plot_it=False, pxsca=65) -> np.ndarray:
    """
    Function to perform continous wavelet transformation with pywavelets library.
    https://pywavelets.readthedocs.io/en/latest/ref/cwt.html

    Parameters
    ----------
    ldos : list
        List of the LDOS at one symmetry point over an energy range.

    Returns
    -------
    coef : np.ndarray
        This gives the 2D image of the wavelet transform.

    """
    coef, _ = pywt.cwt(ldos, np.arange(1, pxsca + 1), "morl")
    if plot_it:
        cmap = plt.cm.coolwarm.copy()
        plt.imshow(
            coef,
            cmap=cmap,
            interpolation="none",
            extent=[1, 66, pxsca + 1, pxsca + 1],
            aspect="auto",
            # vmax=abs(coef).max(),
            # vmin=-abs(coef).max(),
        )
        plt.colorbar()
        plt.show()

    return coef


# ---------------------------- Prelude - Reading experimental data  -------------------------

dos_cnp = io.loadmat("m_cnp.mat")
dos_nemat = io.loadmat("m_nematic.mat")
dosothers = io.loadmat("maps.mat")
en = io.loadmat("v.mat")

nsamples = 8
dosr = []
sca = []
vectorized_images = []
px = 65
pxsca = 65
scatemp = np.zeros((pxsca, pxsca, 3))  # Scaleogram of 3 stack points
label = np.zeros((nsamples))

# Convert dict to numpy array
# N = 101 (energies between -100-100 meV; resolution = 2 meV)
# Energy data is contained in 'v619' key
# dim(en) = N
en = np.array(en.get("v619"))
ndosr = 4
en[:] = np.round(en[:], ndosr)
# 64 is the initial dimension from the exp data.
dos = np.zeros((64, 64, len(en), nsamples))
# Dos(r) Data close to charge neutrality (no nematicity), contained in 'v619' key
# dim(dos_cnp) = (64,64,N)
dos_cnp = np.array(dos_cnp.get("m619"))
# Dos(r) Data close to half filling (nematicit), contained in 'm428' key
# dim(dos_nemat) = (64,64,N)
dos_nemat = np.array(dos_nemat.get("m428"))

# Create vectors with DOS which will be saved in .npz files.
dosBAAC = np.zeros((len(en)))
dosABCA = np.zeros((len(en)))
dosABAB = np.zeros((len(en)))
# dosothers1-9,13.shape = 64, 64, 101
dosothers1 = np.array(dosothers.get("m358"))
dosothers2 = np.array(dosothers.get("m363"))
dosothers3 = np.array(dosothers.get("m370"))
dosothers4 = np.array(dosothers.get("m372"))
dosothers5 = np.array(dosothers.get("m374"))
dosothers6 = np.array(dosothers.get("m376"))
dosothers7 = np.array(dosothers.get("m378"))
dosothers8 = np.array(dosothers.get("m387"))
dosothers9 = np.array(dosothers.get("m428"))
dosothers13 = np.array(dosothers.get("m619"))
# If you want to add from 10 -> 12 we'll have to interpolate between the
# 51 points and generate again a DOS(w) for 65 points to the scaleogram
dosothers10 = np.array(dosothers.get("m596"))
dosothers11 = np.array(dosothers.get("m598"))
dosothers12 = np.array(dosothers.get("m600"))


# We focus on the 7 fillings from paper []
dos[:, :, :, 0] = np.array(dosothers.get("m376"))
dos[:, :, :, 1] = np.array(dosothers.get("m370"))
dos[:, :, :, 2] = np.array(dosothers.get("m374"))
dos[:, :, :, 3] = np.array(dosothers.get("m619"))
dos[:, :, :, 4] = np.array(dosothers.get("m378"))
dos[:, :, :, 5] = np.array(dosothers.get("m428"))
dos[:, :, :, 6] = np.array(dosothers.get("m358"))
dos[:, :, :, 7] = np.array(dosothers.get("m387"))

# Dictionary and vector with filling fraction of the CFB
d = {
    "m358": 0.61,
    "m363": -0.84,
    "m370": -0.45,  # 2
    "m372": -0.71,
    "m374": -0.32,  # 3
    "m376": -0.58,  # 1
    "m378": 0.34,  # 5
    "m387": 0.67,  # 7
    "m428": 0.47,  # 6
    "m596": 0.21,  # less data
    "m598": 0.08,  # less data
    "m600": -0.05,  # less data
    "m619": 0,  # 4
}

label[0] = -0.58
label[1] = -0.45
label[2] = -0.32
label[3] = 0.00
label[4] = 0.34
label[5] = 0.47
label[6] = 0.61
label[7] = 0.67

# The numbers refer to the specific pixel position in the original images
# corresponding to BAAC, ABAB, ABCA (Average between 5 positions)

# ---------------------------- 1st PART - DOS(w) -------------------------
# The en vector is reversed
en2 = np.arange(0.06, -0.07, -0.002)
# en2 = np.arange(0.1, -0.102, -0.002)
dosBAAC2 = np.zeros((len(en2)))
dosABCA2 = np.zeros((len(en2)))
dosABAB2 = np.zeros((len(en2)))


# Plot average points for obtaining Dos(r)
plot_it = True
sites = np.zeros((len(label), 2, 5, 3))
if plot_it:

    i = 0
    # BAAC
    sites[i, :, 0, 0] = [28, 55]
    sites[i, :, 1, 0] = [50, 19]
    sites[i, :, 2, 0] = [50, 43]
    sites[i, :, 3, 0] = [28, 7]
    sites[i, :, 4, 0] = [28, 31]
    # ABABi
    sites[i, :, 0, 1] = [21, 41]
    sites[i, :, 1, 1] = [21, 18]
    sites[i, :, 2, 1] = [43, 6]
    sites[i, :, 3, 1] = [42, 31]
    sites[i, :, 4, 1] = [42, 53]
    # ABCAi
    sites[i, :, 0, 2] = [54, 30]
    sites[i, :, 1, 2] = [54, 53]
    sites[i, :, 2, 2] = [33, 42]
    sites[i, :, 3, 2] = [36, 18]
    sites[i, :, 4, 2] = [14, 30]
    plt.imshow(dos[:, :, 44, 1], cmap="inferno")
    plt.colorbar()

    # BAAC purple
    plt.scatter(
        sites[i, 0, 0, 0],
        sites[i, 1, 0, 0],
        marker="p",
        s=100,
        c="purple",
        edgecolors="black",
    )
    plt.scatter(
        sites[i, 0, 1, 0],
        sites[i, 1, 1, 0],
        marker="p",
        s=100,
        c="purple",
        edgecolors="black",
    )
    plt.scatter(
        sites[i, 0, 2, 0],
        sites[i, 1, 2, 0],
        marker="p",
        s=100,
        c="purple",
        edgecolors="black",
    )
    plt.scatter(
        sites[i, 0, 3, 0],
        sites[i, 1, 3, 0],
        marker="p",
        s=100,
        c="purple",
        edgecolors="black",
    )
    plt.scatter(
        sites[i, 0, 4, 0],
        sites[i, 1, 4, 0],
        marker="p",
        s=100,
        c="purple",
        edgecolors="black",
    )

    # ABAB red
    plt.scatter(
        sites[i, 0, 0, 1],
        sites[i, 1, 0, 1],
        marker="p",
        s=100,
        c="red",
        edgecolors="black",
    )
    plt.scatter(
        sites[i, 0, 1, 1],
        sites[i, 1, 1, 1],
        marker="p",
        s=100,
        c="red",
        edgecolors="black",
    )
    plt.scatter(
        sites[i, 0, 2, 1],
        sites[i, 1, 2, 1],
        marker="p",
        s=100,
        c="red",
        edgecolors="black",
    )
    plt.scatter(
        sites[i, 0, 3, 1],
        sites[i, 1, 3, 1],
        marker="p",
        s=100,
        c="red",
        edgecolors="black",
    )
    plt.scatter(
        sites[i, 0, 4, 1],
        sites[i, 1, 4, 1],
        marker="p",
        s=100,
        c="red",
        edgecolors="black",
    )

    # ABCA black
    plt.scatter(
        sites[i, 0, 0, 2],
        sites[i, 1, 0, 2],
        marker="p",
        s=100,
        c="black",
        edgecolors="green",
    )
    plt.scatter(
        sites[i, 0, 1, 2],
        sites[i, 1, 1, 2],
        marker="p",
        s=100,
        c="black",
        edgecolors="green",
    )
    plt.scatter(
        sites[i, 0, 2, 2],
        sites[i, 1, 2, 2],
        marker="p",
        s=100,
        c="black",
        edgecolors="green",
    )
    plt.scatter(
        sites[i, 0, 3, 2],
        sites[i, 1, 3, 2],
        marker="p",
        s=100,
        c="black",
        edgecolors="green",
    )
    plt.scatter(
        sites[i, 0, 4, 2],
        sites[i, 1, 4, 2],
        marker="p",
        s=100,
        c="black",
        edgecolors="green",
    )
    # plt.axis("off")
    plt.tight_layout()
    plt.savefig("sample_points.pdf", dpi=300)
    # plt.show()

for channel in range(len(label)):

    dosBAAC[:] = (
        dos[55, 28, :, channel]
        + dos[19, 50, :, channel]
        + dos[43, 50, :, channel]
        + dos[7, 28, :, channel]
        + dos[31, 28, :, channel]
    ) / 5
    dosABAB[:] = (
        dos[41, 21, :, channel]
        + dos[18, 21, :, channel]
        + dos[6, 43, :, channel]
        + dos[31, 42, :, channel]
        + dos[53, 42, :, channel]
    ) / 5
    dosABCA[:] = (
        dos[30, 54, :, channel]
        + dos[53, 54, :, channel]
        + dos[42, 33, :, channel]
        + dos[18, 36, :, channel]
        + dos[30, 14, :, channel]
    ) / 5

    if plot_it_full:
        fig, ax = plt.subplots()
        a1 = np.max(dosBAAC)
        a2 = np.max(dosABCA)
        a3 = np.max(dosABAB)
        textstr = "\n".join((r"$n_{s}=%.2f$" % (label[channel],),))

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

        # place a text box in upper left in axes coords
        ax.text(
            0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=props,
        )
        ax.tick_params(axis="x", which="minor", bottom=False)
        ax.tick_params(axis="y", which="minor", bottom=False)

        plt.plot(en[:], dosBAAC[:] / a1 + 1, "--bo", c="purple", label="BAAC")
        plt.plot(en[:], dosABAB[:] / a2 + 2, "--bo", c="red", label="ABAB")
        plt.plot(en[:], dosABCA[:] / a3, "--bo", c="black", label="ABCA")
        # plt.xlim(-0.07, 0.06)
        plt.legend()
        plt.show()

    # Convert intial Dos(w) to 65x65 scaleograms
    p = 0
    for i in range(len(en)):
        for j in range(len(en2)):
            if np.abs(en[i] - en2[j]) < 0.001:
                dosBAAC2[p] = dosBAAC[i]
                dosABCA2[p] = dosABCA[i]
                dosABAB2[p] = dosABAB[i]
                p += 1

    if plot_it_partial:
        textstr = "\n".join((r"$n_{s}=%.2f$" % (label[channel],),))

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        fig, ax = plt.subplots()
        a1 = np.max(dosBAAC2)
        a2 = np.max(dosABCA2)
        a3 = np.max(dosABAB2)

        textstr = "\n".join((r"$n_{s}=%.2f$" % (label[channel],),))

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

        # place a text box in upper left in axes coords
        ax.text(
            0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=props,
        )
        ax.tick_params(which="both", width=2.5, direction="in")
        # ax.tick_params(axis="x", which="minor", bottom=False)
        # ax.tick_params(axis="y", which="minor", bottom=False)
        ax.tick_params(which="major", length=7)
        ax.tick_params(which="minor", length=4)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.minorticks_on()
        font_size = 16  # Adjust as appropriate.
        plt.xlabel(r"Energy $\omega$ (eV)", fontsize=font_size)
        plt.ylabel(r"LDOS (a.u.)", fontsize=font_size)
        plt.plot(en2[:], dosBAAC2[:] / a1 + 1, "--bo", c="purple", label="BAAC")
        plt.plot(en2[:], dosABAB2[:] / a2 + 2, "--bo", c="red", label="ABAB")
        plt.plot(en2[:], dosABCA2[:] / a3, "--bo", c="black", label="ABCA")
        plt.xlim(min(en2), max(en2))
        plt.legend()
        plt.tight_layout()
        plt.savefig("partialplot.pdf", dpi=300)
        plt.show()

    # The scaleograms in the trained dataset go from negative to positive energies.
    # Here it's the opposite, so we need to change it for consistency.
    print(channel)
    scatemp[:, :, 0] = wavelet_trafo(dosBAAC2[::-1])
    scatemp[:, :, 1] = wavelet_trafo(dosABCA2[::-1])
    scatemp[:, :, 2] = wavelet_trafo(dosABAB2[::-1])

    for j in range(3):
        sca.append(scatemp[:, :, j])


# 2ND PART - DOS(R)


def dos_processing(data: np.ndarray, px=65) -> np.ndarray:
    """
    TD: show flags to add contrast or smooth without modifying the function
    Function to smooth, change resolution and increase contrast of DOS(r) images.
    https://pywavelets.readthedocs.io/en/latest/ref/cwt.html

    Parameters
    ----------
    data : np.ndarray
        Array with (4, 64, 64) dimension, containing 64 x 64 experimental DOS(r) images for
        RV1, VFB, CFB, RC1
    px : integer
        Pixel size of DOS(r) images

    Returns
    -------
    dataf : np.ndarray
        Array with shape (4, 65, 65) containing the images  {RV1, VFB, CFB, RC1}

    """
    dataf = np.zeros((4, px, px))
    for j in range(4):
        datatemp = np.array(data[j, :, :])
        # 1) Saving the DOS(r) as an image. In this stage we convert the 64x64 to a 244x244 resolution
        # Saving the image to a .png also smooths it locally

        # a colormap and a normalization instance
        cmap = plt.cm.inferno
        norm = plt.Normalize(vmin=datatemp.min(), vmax=datatemp.max())
        # map the normalized datatemp to colors
        # image is now RGBA (512x512x4)
        image = cmap(norm(datatemp))
        # Remove frame of image
        w = h = 1
        fig = plt.figure(frameon=False)
        fig.set_size_inches(w, h)
        # Fill the whole figure
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(image, aspect="auto")
        plt.savefig(f"temp{j}.png", bbox_inches="tight", pad_inches=0, dpi=355)
        plt.clf()
        plt.cla()
        plt.close()

        # 2) Load the image, and add more contrast to ressemble colors of training dataset
        image_path = f"temp{j}.png"
        # a = np.arkray(Image.open("temp.png"))
        rgb_img = cv2.imread(image_path)
        # convert from RGB color-space to YCrCb
        ycrcb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(ycrcb_img)
        img = ycrcb_img
        # Blur image to remove some noies/defect

        sigma = 1
        img = skimage.filters.gaussian(
            img, sigma=(sigma, sigma), truncate=3.5, channel_axis=2
        )

        # Cropping image to size consistent in training dataset
        # dostemp final shape = 65x65
        # dostemp = st.resize(img[20:310, 70:360], (px, px))
        dostemp = st.resize(img[0:300, 50:350], (px, px))
        # dostemp = st.resize(img[0:350, 0:350], (px, px))

        # Saving new map
        dataf[j, :, :] = dostemp

    return dataf[:, :, :]


def ind_en(array: np.ndarray, val: float) -> int:
    """
    TD: Maybe add some try error here
    Function that returns index of element with certain value in array.

    Parameters
    ----------
    array : np.ndarray
        Vector with energies
    val : float
        energy value

    """
    j = np.where(array == val)

    # Check if first array from np.where is [] or some valid index

    if j[0].size:
        j = int(j[0])
    else:
        # print("")
        print(
            "Energy not found in array. Check if val and array are valid values, and be sure that val is in array."
        )
        # print(sys.stderr, "Energy not found in array. Check if val and array are valid values, and be sure that val is in array")
        # print(sys.stderr, "Exception: %s" % str(e))
        sys.exit(1)
    return j


# data contais RV1, VFB, CFB and RC1

index = np.zeros((len(label), 4), int)

# This has to be done explicitly since
# doping changes the energy label for bands
# See Fig. 2a) from ref. [1]

# doping -0.58 n_s, displacement field -0.13 V nm^{-1}
ind_rv1 = ind_en(en, -0.056)
ind_vfb = ind_en(en, -0.008)
ind_cfb = ind_en(en, 0.010)
ind_rc1 = ind_en(en, 0.056)
index[0, :] = np.array([ind_rv1, ind_vfb, ind_cfb, ind_rc1])

# doping -0.45 n_s, displacement field -0.10 V nm^{-1}
ind_rv1 = ind_en(en, -0.058)
ind_vfb = ind_en(en, -0.006)
ind_cfb = ind_en(en, 0.006)
ind_rc1 = ind_en(en, 0.052)
index[1, :] = np.array([ind_rv1, ind_vfb, ind_cfb, ind_rc1])

# doping -0.32 n_s, displacement field -0.07 V nm^{-1}
ind_rv1 = ind_en(en, -0.060)
ind_vfb = ind_en(en, -0.006)
ind_cfb = ind_en(en, 0.004)
ind_rc1 = ind_en(en, 0.052)
index[2, :] = np.array([ind_rv1, ind_vfb, ind_cfb, ind_rc1])

# doping 0 n_s, displacement field 0 V nm^{-1}
ind_rv1 = ind_en(en, -0.058)
ind_vfb = ind_en(en, -0.008)
ind_cfb = ind_en(en, 0.002)
ind_rc1 = ind_en(en, 0.050)
index[3, :] = np.array([ind_rv1, ind_vfb, ind_cfb, ind_rc1])

# doping 0.34 n_s, displacement field 0.08 V nm^{-1}
ind_rv1 = ind_en(en, -0.064)
ind_vfb = ind_en(en, -0.010)
ind_cfb = ind_en(en, 0.000)
ind_rc1 = ind_en(en, 0.046)
index[4, :] = np.array([ind_rv1, ind_vfb, ind_cfb, ind_rc1])

# doping 0.47 n_s, displacement field 0.11 V nm^{-1}
ind_rv1 = ind_en(en, -0.064)
ind_vfb = ind_en(en, -0.012)
ind_cfb = ind_en(en, 0.002)
ind_rc1 = ind_en(en, 0.044)
index[5, :] = np.array([ind_rv1, ind_vfb, ind_cfb, ind_rc1])

# doping 0.61 n_s
ind_rv1 = ind_en(en, -0.066)
ind_vfb = ind_en(en, -0.018)
ind_cfb = ind_en(en, -0.004)
ind_rc1 = ind_en(en, 0.04)
index[6, :] = np.array([ind_rv1, ind_vfb, ind_cfb, ind_rc1])

# doping 0.67 n_s, displacement field 0.16 V nm^{-1}
ind_rv1 = ind_en(en, -0.068)
ind_vfb = ind_en(en, -0.012)
ind_cfb = ind_en(en, -0.004)
ind_rc1 = ind_en(en, 0.040)
index[7, :] = np.array([ind_rv1, ind_vfb, ind_cfb, ind_rc1])

dataf = np.zeros((4, px, px))
# index = index()
# for j in range(len(label)):

for j in range(len(label)):

    data = np.array(
        [
            dos[:, :, index[j, 0], j],
            dos[:, :, index[j, 1], j],
            dos[:, :, index[j, 2], j],
            dos[:, :, index[j, 3], j],
        ]
    )

    dataf = dos_processing(data)
    dataf[0, :, :] = dataf[0, :, ::-1]
    dataf[1, :, :] = dataf[1, ::-1, ::-1]
    dataf[2, :, :] = dataf[2, ::-1, ::-1]
    dataf[3, :, :] = dataf[3, :, ::-1]

    for p in range(4):
        dosr.append(dataf[p, :, :])

    if plot_it_post:
        f, ax = plt.subplots(2, 4)
        f.suptitle(
            r"$n_{s}=%.2f$" % (label[j],),
        )

        ax[0, 0].imshow(data[0, :, :], cmap="inferno")
        ax[0, 1].imshow(data[1, :, :], cmap="inferno")
        ax[0, 2].imshow(data[2, :, :], cmap="inferno")
        ax[0, 3].imshow(data[3, :, :], cmap="inferno")
        ax[1, 0].imshow(dataf[0, :, :], cmap="inferno")
        ax[1, 1].imshow(dataf[1, :, :], cmap="inferno")
        ax[1, 2].imshow(dataf[2, :, :], cmap="inferno")
        ax[1, 3].imshow(dataf[3, :, :], cmap="inferno")
        [axi.set_axis_off() for axi in ax.ravel()]
        plt.show()
        plt.clf()
        plt.cla()
        plt.close()

np.savez(
    "expdata.npz",
    DataX=dosr,
    DataZ=sca,
    DataP=label,
)

print(np.array(dosr).shape)
print(np.array(sca).shape)
plot_final = False
if plot_final:
    dosr = np.array(dosr)
    print(len(dosr))
    nsamples = 8
    matplotlib.rcParams["axes.linewidth"] = 1.4  # width of frames
    f, ax = plt.subplots(nsamples, 4, constrained_layout=True)

    j = 0
    # for i in range(len(dosr)-1, 0, -4):
    for i in range(0, len(dosr), 4):
        print(i)

        ax[j, 0].imshow(dosr[i, :, :], cmap="inferno")
        ax[j, 1].imshow(dosr[i + 1, :, :], cmap="inferno")
        ax[j, 2].imshow(dosr[i + 2, :, :], cmap="inferno")
        ax[j, 3].imshow(dosr[i + 3, :, :], cmap="inferno")
        # ax[j, 3].imshow(dosr[i, :, :], cmap="inferno")
        # ax[j, 2].imshow(dosr[i - 1, :, :], cmap="inferno")
        # ax[j, 1].imshow(dosr[i - 2, :, :], cmap="inferno")
        # ax[j, 0].imshow(dosr[i - 3, :, :], cmap="inferno")
        [axi.get_yaxis().set_ticks([]) for axi in ax.ravel()]
        [axi.get_xaxis().set_ticks([]) for axi in ax.ravel()]
        j += 1

    # plt.show()
    plt.tight_layout()
    plt.savefig("plot_7_dosr.pdf", dpi=300)
