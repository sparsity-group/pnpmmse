import os
import numpy as np
import scico
from scico import functional, linop, loss
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim
import odl
from utils import measure, load_dataset
NOISE_STD = 2.0
DATA_ROOT = "./mayoct"


def get_operators(space_range, img_size, num_angles, det_shape):
    ##############compute projection#################
    space = odl.uniform_discr(
        [-space_range, -space_range],
        [space_range, space_range],
        (img_size, img_size),
        dtype="float32",
        weighting=1.0,
    )
    geometry = odl.tomo.geometry.parallel.parallel_beam_geometry(
        space, num_angles=num_angles, det_shape=det_shape
    )

    fwd_op_odl = odl.tomo.RayTransform(
        space, geometry, impl="astra_cuda"
    )  # astra_cuda, skimage
    fbp_op_odl = odl.tomo.fbp_op(fwd_op_odl)
    adjoint_op_odl = fwd_op_odl.adjoint

    def fwd_op_numpy(x):
        return fwd_op_odl(np.array(x)).asarray().astype(np.float32)

    def adjoint_op_numpy(x):
        return adjoint_op_odl(np.array(x)).asarray().astype(np.float32)

    def fbp_op_numpy(x):
        return fbp_op_odl(np.array(x)).asarray().astype(np.float32)
    return fwd_op_numpy, adjoint_op_numpy, fbp_op_numpy

