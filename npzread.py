import numpy as np
import matplotlib.pyplot as plt
from scipy import io
import numpy as np
import sys

import cv2
from PIL import Image, ImageFilter

# import PIL.Image
import skimage.transform as st
import pywt

# Reference
# [1] Microscopic nematicity: Rubio-Verdú, C., Turkel, S., Song, Y. et al. Nat. Phys. 18, 196–202 (2022). "Moiré nematic phase in twisted double bilayer graphene.


def wavelet_trafo(ldos: list) -> np.ndarray:
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
    coef, _ = pywt.cwt(ldos, np.arange(1, 66), "morl")
    cmap = plt.cm.coolwarm.copy()
    plt.imshow(
        coef,
        cmap=cmap,
        interpolation="none",
        extent=[1, 66, 1, 66],
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

dosr = []
sca = []
vectorized_images = []
px = 65
scatemp = np.zeros((px, px, 3))  # Scaleogram of 3 stack points
label = np.zeros((7))

# Convert dict to numpy array
# N = 101 (energies between -100-100 meV; resolution = 2 meV)
# Energy data is contained in 'v619' key
# dim(en) = N
en = np.array(en.get("v619"))
en[:] = np.round(en[:], 4)
dos = np.zeros((64, 64, len(en), 7))
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
# If you want to add from 10 -> 13 we'll have to interpolate between the
# 51 points and generate again a DOS(w) for 65 points to the scaleogram
dosothers10 = np.array(dosothers.get("m596"))
dosothers11 = np.array(dosothers.get("m598"))
dosothers12 = np.array(dosothers.get("m600"))


dos[:, :, :, 0] = np.array(dosothers.get("m376"))
dos[:, :, :, 1] = np.array(dosothers.get("m370"))
dos[:, :, :, 2] = np.array(dosothers.get("m374"))
dos[:, :, :, 3] = np.array(dosothers.get("m619"))
dos[:, :, :, 4] = np.array(dosothers.get("m378"))
dos[:, :, :, 5] = np.array(dosothers.get("m428"))
dos[:, :, :, 6] = np.array(dosothers.get("m387"))

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
    "m596": 0.21,
    "m598": 0.08,
    "m600": -0.05,
    "m619": 0,  # 4
}

label[0] = -0.58
label[1] = -0.45
label[2] = -0.32
label[3] = 0.00
label[4] = 0.34
label[5] = 0.47
label[6] = 0.67

# The numbers refer to the specific pixel position in the original images
# corresponding to BAAC, ABAB, ABCA (Average between 5 positions)

# ---------------------------- 1st PART - DOS(w) -------------------------
# The en vector is reversed
en2 = np.arange(0.06, -0.07, -0.002)
dosBAAC2 = np.zeros((len(en2)))
dosABCA2 = np.zeros((len(en2)))
dosABAB2 = np.zeros((len(en2)))


# Plot average points for obtaining Dos(r)
plot_it = False
if plot_it:

    plt.imshow(dos[:, :, 51, 5], cmap="viridis")
    plt.colorbar()

    # BAAC purple
    plt.scatter(28, 55, marker="p", s=100, c="purple", edgecolors="black")
    plt.scatter(50, 19, marker="p", s=100, c="purple", edgecolors="black")
    plt.scatter(50, 43, marker="p", s=100, c="purple", edgecolors="black")
    plt.scatter(28, 7, marker="p", s=100, c="purple", edgecolors="black")
    plt.scatter(28, 31, marker="p", s=100, c="purple", edgecolors="black")

    # ABAB red
    plt.scatter(21, 41, marker="p", s=100, c="red", edgecolors="black")
    plt.scatter(21, 18, marker="p", s=100, c="red", edgecolors="black")
    plt.scatter(43, 6, marker="p", s=100, c="red", edgecolors="black")
    plt.scatter(42, 31, marker="p", s=100, c="red", edgecolors="black")
    plt.scatter(42, 53, marker="p", s=100, c="red", edgecolors="black")

    # ABCA black
    plt.scatter(54, 30, marker="p", s=100, c="black", edgecolors="green")
    plt.scatter(54, 53, marker="p", s=100, c="black", edgecolors="green")
    plt.scatter(33, 42, marker="p", s=100, c="black", edgecolors="green")
    plt.scatter(36, 18, marker="p", s=100, c="black", edgecolors="green")
    plt.scatter(14, 30, marker="p", s=100, c="black", edgecolors="green")
    plt.axis("off")
    plt.show()

plot_it_full = False
plot_it_partial = False
for channel in range(len(label)):
    channel = 3
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
        ax.tick_params(axis="x", which="minor", bottom=False)
        ax.tick_params(axis="y", which="minor", bottom=False)
        plt.plot(en2[:], dosBAAC2[:] / a1 + 1, "--bo", c="purple", label="BAAC")
        plt.plot(en2[:], dosABAB2[:] / a2 + 2, "--bo", c="red", label="ABAB")
        plt.plot(en2[:], dosABCA2[:] / a3, "--bo", c="black", label="ABCA")
        plt.xlim(-0.07, 0.06)
        plt.legend()
        plt.show()

        # wavelet_trafo(dosBAAC[::-1])
        # wavelet_trafo(dosABCA[::-1])
        # wavelet_trafo(dosABAB[::-1])
        # The scaleograms in the trained dataset go from negative to positive energies.
        # Here it's the opposite, so we need to change it for consistency.
        scatemp[:, :, 0] = wavelet_trafo(dosBAAC2[::-1])
        scatemp[:, :, 1] = wavelet_trafo(dosABCA2[::-1])
        scatemp[:, :, 2] = wavelet_trafo(dosABAB2[::-1])

    for j in range(3):
        sca.append(scatemp[:, :, j])

print(np.asarray(sca).shape)


# 2ND PART - DOS(R)


def dos_processing(data: np.ndarray, px=65) -> np.ndarray:
    """
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
        # Cropping image to size consistent in training dataset
        # dostemp final shape = 65x65
        dostemp = st.resize(img[0:300, 50:350], (px, px))

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

# doping 0.67 n_s, displacement field 0.16 V nm^{-1}
ind_rv1 = ind_en(en, -0.068)
ind_vfb = ind_en(en, -0.012)
ind_cfb = ind_en(en, -0.004)
ind_rc1 = ind_en(en, 0.040)
index[6, :] = np.array([ind_rv1, ind_vfb, ind_cfb, ind_rc1])

dataf = np.zeros((4, px, px))
# index = index()
# for j in range(len(label)):
plot_it_post = False
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

    # plt.figure(figsize=(10, 10))  # specifying the overall grid size

    if plot_it_post:
        f, ax = plt.subplots(2, 4)
        f.suptitle(r"$n_{s}=%.2f$" % (label[j],),)
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
    for p in range(len(dataf)):
        dosr.append(dataf[p, :, :])

print(np.asarray(dosr).shape)
# Save the experimental dataset

np.savez(
    "expdata.npz",
    DataX=dosr,
    DataZ=sca,
    dataP=label,
)

#
# new_image = cv2.medianBlur(image_path, figure_size)
# plt.figure(figsize=(11, 6))
# plt.subplot(121)
# plt.imshow(cv2.cvtColor(image_path, cv2.COLOR_HSV2RGB))
# plt.title("Original")
# plt.xticks([])
# plt.yticks([])
# plt.subplot(122)
# plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB))
# plt.title("Median Filter")
# plt.xticks([])
# plt.yticks([])
# plt.show()
# plt.imshow(b, cmap="inferno")
# plt.colorbar()
# plt.axis("off")
# plt.title("Energy = " + str(np.round(en[i], 3)) + "eV")
# plt.show()
# # dosothers12[:,:, i]= cv2.equalizeHist(dosothers12[:,:, i])
# # plt.savefig(f"/home/jass/Downloads/datasetstrain/same padding/expdata/resultsnemat/dosnemat_en_{a}.pdf")
# # plt.savefig(f"/home/jass/Downloads/datasetstrain/same padding/expdata/results/dos_en_{en[i]}.pdf")
# # plt.savefig(f"/home/jass/Downloads/datasetstrain/same padding/expdata/resultsother/12/dos_en_{en[i]}.pdf")
# plt.clf()
