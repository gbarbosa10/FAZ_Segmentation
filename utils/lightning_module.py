import lightning.pytorch as pl
import torch.nn.functional as F
from torchmetrics.segmentation import HausdorffDistance
from torchmetrics import Accuracy
import torch
import os
import torch.nn as nn
import time
import numpy as np
from torch.utils.data import DataLoader, Dataset
#from torchvision import io #read_image
import lightning as L
import torchvision
import matplotlib.axes as Ax
import torchmetrics
#from torchmetrics.classification import Dice
#from torchmetrics.detection import IntersectionOverUnion
from torchmetrics.regression import PearsonCorrCoef
from torchmetrics.classification import JaccardIndex
from torchmetrics.functional.classification import multilabel_jaccard_index
import statsmodels.api as sm
import cv2 as cv
import math
from skimage import metrics
import pandas as pd
import matplotlib.pyplot as plt
import utils
from skimage import measure
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from skimage.filters import threshold_mean, rank
#from torchvision.utils import draw_segmentation_masks
import torchvision.transforms as T
from torchvision.transforms import ToPILImage
import sys
from skimage.feature import peak_local_max
from skimage.morphology import flood, flood_fill, skeletonize
from torchmetrics import Metric
from torch import Tensor
from glob import glob
from scipy import ndimage
from . import lookup_tables
from torchmetrics.image import PeakSignalNoiseRatio, TotalVariation, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
torch.set_printoptions(profile="full")

def write_img_name_list(y, y_hat, img_dir, text_name):
    
    if np.where(y.int()==y_hat.int())[0].any():
        elem_list = np.argwhere(y.int()!=y_hat.int())[0]
        list_ = []
        for elem in elem_list:
            image = torchvision.io.read_image(img_dir[elem])
            torchvision.utils.save_image(image.float(), text_name + str(elem) + ".jpeg")

def get_largest_contour(contour_list):
    cnt_len = 0
    position = 0
    for num, cnt in enumerate(contour_list):
        if cnt_len < cv.arcLength(cnt,True):
            position = num
        cnt_len = cv.arcLength(cnt,True)
    return contour_list[position]

def get_LargsetContArea(image, imsize_1, imsize_2):
     color = 1
     image_flooded = image.copy()*255
     while np.argwhere(image_flooded==255).size != 0:
         index_list = np.argwhere(image_flooded==255)
         image_flooded = flood_fill(image_flooded, (index_list[0][0], index_list[0][1]), color, connectivity=1)
         color = color +1
     area = 0
     res_im = np.zeros((imsize_1, imsize_2))
     num_right = 0
     for num in range(color): 
         if area < len(np.argwhere(image_flooded==num + 1)):
             area = len(np.argwhere(image_flooded==num + 1))
             num_right = num + 1
     res_im[image_flooded==num_right] = 1

     return res_im

def save_image_batch_3chan(im_tensor, og_im, filepath, lb_list, lr, batch_size):
    transform = T.ToPILImage()
    for num in range(lb_list.shape[0]):
        im = im_tensor[num, :, :, :].reshape(im_tensor.shape[2], im_tensor.shape[3], 1)
        image = torch.cat((im, im, im), 2)
        image= image.cpu().numpy().astype(np.uint8)
        threshold_global_otsu = threshold_mean(im.numpy()[:,:,0]*255)
        global_otsu = im.numpy()[:,:,0]*255 >= threshold_global_otsu    #global_otsu.astype(np.uint8)
        contours, hierarchy = cv.findContours(global_otsu.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        cnt = get_largest_contour(contours)
        
        og_im_ = np.array(transform(torch.tensor(torch.clamp(og_im[num, :, :, :]*255, min=0, max=255), dtype=torch.uint8)))
        segmented_im_1 = draw_mask(og_im_, torch.tensor(lb_list[num, 0:1, :, :]*255, dtype=torch.uint8).reshape(im_tensor.shape[2], im_tensor.shape[3], 1).cpu().numpy(), np.array([0,255,0], dtype='uint8'))
        segmented_im_2 = draw_mask(og_im_, torch.tensor(im_tensor[num, 0:1, :, :]*255, dtype=torch.uint8).reshape(im_tensor.shape[2], im_tensor.shape[3], 1).cpu().numpy(), np.array([255,0,0], dtype='uint8')) 
        segmented_im = np.concatenate((segmented_im_1, segmented_im_2), 1)
        cv.drawContours(image, cnt, -1, (0, 255, 0), 3)
        segmented_im = np.concatenate((segmented_im, image), 1)
        cv.imwrite(filepath + "_3_image_joint_"+ str(num) + "_" + str(lr) + "_" + str(batch_size) + ".png", segmented_im)

def save_image_batch_3chan_pre_train(im_tensor, in_im, lb, filepath, lr, batch_size):
    transform = T.ToPILImage()
    for num in range(in_im.shape[0]):
        #im = im_tensor[num, :, :, :] #.reshape(im_tensor.shape[2], im_tensor.shape[3], 1)
        image = torch.cat((im_tensor[num, :, :, :], im_tensor[num, :, :, :], im_tensor[num, :, :, :]), 0)
        image = np.array(transform(torch.tensor(torch.clamp(image*255, min=0, max=255), dtype=torch.uint8))) #image.cpu().numpy().astype(np.uint8)
        og_im_ = np.array(transform(torch.tensor(torch.clamp(in_im[num, :, :, :]*255, min=0, max=255), dtype=torch.uint8)))
        lb_im_ = np.array(transform(torch.tensor(torch.clamp(lb[num, :, :, :]*255, min=0, max=255), dtype=torch.uint8)))
        joint_im = np.concatenate((image, og_im_, lb_im_), 1)
        cv.imwrite(filepath + "_2_image_joint_"+ str(num) + "_" + str(lr) + "_" + str(batch_size) + ".png", joint_im)

def draw_mask(image, mask_generated, color_array) :
    masked_image = image.copy()
    masked_image = np.where(mask_generated.astype(np.uint8),
                            color_array,
                            masked_image)
    
    masked_image = masked_image.astype(np.uint8)
    return cv.addWeighted(image, 0.3, masked_image, 0.7, 0)

def area_calculation(oct_image):
    area_list = torch.Tensor([0]).reshape(1).cuda()
    for num in range(oct_image.shape[0]):
        
        im = oct_image[num, :, :, :].cuda()
        oct_image_out = torch.where(im >= 0.5, 1 , 0).cuda()
        area_list = torch.concat((area_list, torch.Tensor(torch.sum(torch.sum(oct_image_out))).reshape(1)), dim=0).cuda()
        
    return torch.Tensor(area_list[1:])

def perimiter_calculation(oct_image):
    perimiter_list = torch.Tensor([0]).reshape(1)#.cuda()
    for num in range(oct_image.shape[0]):
        im = oct_image[num, :, :, :].reshape(oct_image.shape[2], oct_image.shape[3], 1)

        threshold_global_otsu = threshold_mean(im.detach().cpu().numpy()[:,:,0]*255)
        global_otsu = im.detach().cpu().numpy()[:,:,0]*255 >= threshold_global_otsu
        
        contours, hierarchy = cv.findContours(global_otsu.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        
        cnt = get_largest_contour(contours)
        perimeter = cv.arcLength(cnt,True)
        perimiter_list = torch.concat((perimiter_list, torch.Tensor([perimeter]).reshape(1)), dim=0)
    return torch.Tensor(perimiter_list[1:])

def roundness_calculation(oct_image):
    roundness_list = torch.Tensor([0]).reshape(1).cpu()
    for num in range(oct_image.shape[0]):
        im = oct_image[num, :, :, :].reshape(oct_image.shape[2], oct_image.shape[3], 1)
        thresh = torch.where(im >= 0.5, 1 , 0)
        contours,hierarchy = cv.findContours(thresh.cpu().numpy().astype(np.uint8), 1, 2)
        if len(contours)==0:
            cnt = np.array([[[ 45, 188]], [[ 45, 189]], [[ 50, 194]],[[ 52, 194]],[[ 52, 193]],[[ 47, 188]]]).astype(np.int32) 
        else:
            cnt = contours[0]
        perimeter = cv.arcLength(cnt,True)
        roundness = (4*math.pi*torch.Tensor(torch.sum(torch.sum(im))).reshape(1))/(perimeter*perimeter)
        roundness_list = torch.concat((roundness_list, torch.Tensor([roundness.cpu()]).reshape(1)), dim=0)
    return torch.Tensor(roundness_list[1:]) 

# --------------------------------------------------- Hausdorff Distance Auxilary Functions -------------------------------------------------------
# https://github.com/google-deepmind/surface-distance/tree/master
def _assert_is_numpy_array(name, array):
  """Raises an exception if `array` is not a numpy array."""
  if not isinstance(array, np.ndarray):
    raise ValueError("The argument {!r} should be a numpy array, not a "
                     "{}".format(name, type(array)))

def _assert_is_bool_numpy_array(name, array):
  _assert_is_numpy_array(name, array)
  if array.dtype != bool:
    raise ValueError("The argument {!r} should be a numpy array of type bool, "
                     "not {}".format(name, array.dtype))

def _check_nd_numpy_array(name, array, num_dims):
  """Raises an exception if `array` is not a `num_dims`-D numpy array."""
  if len(array.shape) != num_dims:
    raise ValueError("The argument {!r} should be a {}D array, not of "
                     "shape {}".format(name, num_dims, array.shape))

def _check_2d_numpy_array(name, array):
  _check_nd_numpy_array(name, array, num_dims=2)

def _check_3d_numpy_array(name, array):
  _check_nd_numpy_array(name, array, num_dims=3)

def _compute_bounding_box(mask):
  """Computes the bounding box of the masks.

  This function generalizes to arbitrary number of dimensions great or equal
  to 1.

  Args:
    mask: The 2D or 3D numpy mask, where '0' means background and non-zero means
      foreground.

  Returns:
    A tuple:
     - The coordinates of the first point of the bounding box (smallest on all
       axes), or `None` if the mask contains only zeros.
     - The coordinates of the second point of the bounding box (greatest on all
       axes), or `None` if the mask contains only zeros.
  """
  num_dims = len(mask.shape)
  bbox_min = np.zeros(num_dims, np.int64)
  bbox_max = np.zeros(num_dims, np.int64)

  # max projection to the x0-axis
  proj_0 = np.amax(mask, axis=tuple(range(num_dims))[1:])
  idx_nonzero_0 = np.nonzero(proj_0)[0]
  if len(idx_nonzero_0) == 0:  # pylint: disable=g-explicit-length-test
    return None, None

  bbox_min[0] = np.min(idx_nonzero_0)
  bbox_max[0] = np.max(idx_nonzero_0)

  # max projection to the i-th-axis for i in {1, ..., num_dims - 1}
  for axis in range(1, num_dims):
    max_over_axes = list(range(num_dims))  # Python 3 compatible
    max_over_axes.pop(axis)  # Remove the i-th dimension from the max
    max_over_axes = tuple(max_over_axes)  # numpy expects a tuple of ints
    proj = np.amax(mask, axis=max_over_axes)
    idx_nonzero = np.nonzero(proj)[0]
    bbox_min[axis] = np.min(idx_nonzero)
    bbox_max[axis] = np.max(idx_nonzero)

  return bbox_min, bbox_max

def _crop_to_bounding_box(mask, bbox_min, bbox_max):
  """Crops a 2D or 3D mask to the bounding box specified by `bbox_{min,max}`."""
  # we need to zeropad the cropped region with 1 voxel at the lower,
  # the right (and the back on 3D) sides. This is required to obtain the
  # "full" convolution result with the 2x2 (or 2x2x2 in 3D) kernel.
  # TODO:  This is correct only if the object is interior to the
  # bounding box.
  cropmask = np.zeros((bbox_max - bbox_min) + 2, np.uint8)

  num_dims = len(mask.shape)
  # pyformat: disable
  if num_dims == 2:
    cropmask[0:-1, 0:-1] = mask[bbox_min[0]:bbox_max[0] + 1,
                                bbox_min[1]:bbox_max[1] + 1]
  elif num_dims == 3:
    cropmask[0:-1, 0:-1, 0:-1] = mask[bbox_min[0]:bbox_max[0] + 1,
                                      bbox_min[1]:bbox_max[1] + 1,
                                      bbox_min[2]:bbox_max[2] + 1]
  # pyformat: enable
  else:
    assert False

  return cropmask

def _sort_distances_surfels(distances, surfel_areas):
  """Sorts the two list with respect to the tuple of (distance, surfel_area).

  Args:
    distances: The distances from A to B (e.g. `distances_gt_to_pred`).
    surfel_areas: The surfel areas for A (e.g. `surfel_areas_gt`).

  Returns:
    A tuple of the sorted (distances, surfel_areas).
  """
  sorted_surfels = np.array(sorted(zip(distances, surfel_areas)))
  return sorted_surfels[:, 0], sorted_surfels[:, 1]

def compute_surface_distances(mask_gt,
                              mask_pred,
                              spacing_mm):
    """Computes closest distances from all surface points to the other surface.

    This function can be applied to 2D or 3D tensors. For 2D, both masks must be
    2D and `spacing_mm` must be a 2-element list. For 3D, both masks must be 3D
    and `spacing_mm` must be a 3-element list. The description is done for the 2D
    case, and the formulation for the 3D case is present is parenthesis,
    introduced by "resp.".

    Finds all contour elements (resp surface elements "surfels" in 3D) in the
    ground truth mask `mask_gt` and the predicted mask `mask_pred`, computes their
    length in mm (resp. area in mm^2) and the distance to the closest point on the
    other contour (resp. surface). It returns two sorted lists of distances
    together with the corresponding contour lengths (resp. surfel areas). If one
    of the masks is empty, the corresponding lists are empty and all distances in
    the other list are `inf`.

    Args:
        mask_gt: 2-dim (resp. 3-dim) bool Numpy array. The ground truth mask.
        mask_pred: 2-dim (resp. 3-dim) bool Numpy array. The predicted mask.
        spacing_mm: 2-element (resp. 3-element) list-like structure. Voxel spacing
        in x0 anx x1 (resp. x0, x1 and x2) directions.

    Returns:
        A dict with:
        "distances_gt_to_pred": 1-dim numpy array of type float. The distances in mm
            from all ground truth surface elements to the predicted surface,
            sorted from smallest to largest.
        "distances_pred_to_gt": 1-dim numpy array of type float. The distances in mm
            from all predicted surface elements to the ground truth surface,
            sorted from smallest to largest.
        "surfel_areas_gt": 1-dim numpy array of type float. The length of the
        of the ground truth contours in mm (resp. the surface elements area in
        mm^2) in the same order as distances_gt_to_pred.
        "surfel_areas_pred": 1-dim numpy array of type float. The length of the
        of the predicted contours in mm (resp. the surface elements area in
        mm^2) in the same order as distances_gt_to_pred.

    Raises:
        ValueError: If the masks and the `spacing_mm` arguments are of incompatible
        shape or type. Or if the masks are not 2D or 3D.
    """
    # The terms used in this function are for the 3D case. In particular, surface
    # in 2D stands for contours in 3D. The surface elements in 3D correspond to
    # the line elements in 2D.

    _assert_is_bool_numpy_array("mask_gt", mask_gt)
    _assert_is_bool_numpy_array("mask_pred", mask_pred)

    if not len(mask_gt.shape) == len(mask_pred.shape) == len(spacing_mm):
        raise ValueError("The arguments must be of compatible shape. Got mask_gt "
                        "with {} dimensions ({}) and mask_pred with {} dimensions "
                        "({}), while the spacing_mm was {} elements.".format(
                            len(mask_gt.shape),
                            mask_gt.shape, len(mask_pred.shape), mask_pred.shape,
                            len(spacing_mm)))

    num_dims = len(spacing_mm)
    if num_dims == 2:
        _check_2d_numpy_array("mask_gt", mask_gt)
        _check_2d_numpy_array("mask_pred", mask_pred)

        # compute the area for all 16 possible surface elements
        # (given a 2x2 neighbourhood) according to the spacing_mm
        neighbour_code_to_surface_area = (
            lookup_tables.create_table_neighbour_code_to_contour_length(spacing_mm))
        kernel = lookup_tables.ENCODE_NEIGHBOURHOOD_2D_KERNEL
        full_true_neighbours = 0b1111
    elif num_dims == 3:
        _check_3d_numpy_array("mask_gt", mask_gt)
        _check_3d_numpy_array("mask_pred", mask_pred)

        # compute the area for all 256 possible surface elements
        # (given a 2x2x2 neighbourhood) according to the spacing_mm
        neighbour_code_to_surface_area = (
            lookup_tables.create_table_neighbour_code_to_surface_area(spacing_mm))
        kernel = lookup_tables.ENCODE_NEIGHBOURHOOD_3D_KERNEL
        full_true_neighbours = 0b11111111
    else:
        raise ValueError("Only 2D and 3D masks are supported, not "
                        "{}D.".format(num_dims))

    # compute the bounding box of the masks to trim the volume to the smallest
    # possible processing subvolume
    bbox_min, bbox_max = _compute_bounding_box(mask_gt | mask_pred)
    # Both the min/max bbox are None at the same time, so we only check one.
    if bbox_min is None:
        return {
            "distances_gt_to_pred": np.array([]),
            "distances_pred_to_gt": np.array([]),
            "surfel_areas_gt": np.array([]),
            "surfel_areas_pred": np.array([]),
        }

    # crop the processing subvolume.
    cropmask_gt = _crop_to_bounding_box(mask_gt, bbox_min, bbox_max)
    cropmask_pred = _crop_to_bounding_box(mask_pred, bbox_min, bbox_max)

    # compute the neighbour code (local binary pattern) for each voxel
    # the resulting arrays are spacially shifted by minus half a voxel in each
    # axis.
    # i.e. the points are located at the corners of the original voxels
    neighbour_code_map_gt = ndimage.filters.correlate(
        cropmask_gt.astype(np.uint8), kernel, mode="constant", cval=0)
    neighbour_code_map_pred = ndimage.filters.correlate(
        cropmask_pred.astype(np.uint8), kernel, mode="constant", cval=0)

    # create masks with the surface voxels
    borders_gt = ((neighbour_code_map_gt != 0) &
                    (neighbour_code_map_gt != full_true_neighbours))
    borders_pred = ((neighbour_code_map_pred != 0) &
                    (neighbour_code_map_pred != full_true_neighbours))

    # compute the distance transform (closest distance of each voxel to the
    # surface voxels)
    if borders_gt.any():
        distmap_gt = ndimage.morphology.distance_transform_edt(
            ~borders_gt, sampling=spacing_mm)
    else:
        distmap_gt = np.inf * np.ones(borders_gt.shape)

    if borders_pred.any():
        distmap_pred = ndimage.morphology.distance_transform_edt(
            ~borders_pred, sampling=spacing_mm)
    else:
        distmap_pred = np.inf * np.ones(borders_pred.shape)

    # compute the area of each surface element
    surface_area_map_gt = neighbour_code_to_surface_area[neighbour_code_map_gt]
    surface_area_map_pred = neighbour_code_to_surface_area[
        neighbour_code_map_pred]

    # create a list of all surface elements with distance and area
    distances_gt_to_pred = distmap_pred[borders_gt]
    distances_pred_to_gt = distmap_gt[borders_pred]
    surfel_areas_gt = surface_area_map_gt[borders_gt]
    surfel_areas_pred = surface_area_map_pred[borders_pred]

    # sort them by distance
    if distances_gt_to_pred.shape != (0,):
        distances_gt_to_pred, surfel_areas_gt = _sort_distances_surfels(
            distances_gt_to_pred, surfel_areas_gt)

    if distances_pred_to_gt.shape != (0,):
        distances_pred_to_gt, surfel_areas_pred = _sort_distances_surfels(
            distances_pred_to_gt, surfel_areas_pred)

    return {
        "distances_gt_to_pred": distances_gt_to_pred,
        "distances_pred_to_gt": distances_pred_to_gt,
        "surfel_areas_gt": surfel_areas_gt,
        "surfel_areas_pred": surfel_areas_pred,
    }

def compute_robust_hausdorff(surface_distances, percent):
  """Computes the robust Hausdorff distance.

  Computes the robust Hausdorff distance. "Robust", because it uses the
  `percent` percentile of the distances instead of the maximum distance. The
  percentage is computed by correctly taking the area of each surface element
  into account.

  Args:
    surface_distances: dict with "distances_gt_to_pred", "distances_pred_to_gt"
      "surfel_areas_gt", "surfel_areas_pred" created by
      compute_surface_distances()
    percent: a float value between 0 and 100.

  Returns:
    a float value. The robust Hausdorff distance in mm.
  """
  distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
  distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
  surfel_areas_gt = surface_distances["surfel_areas_gt"]
  surfel_areas_pred = surface_distances["surfel_areas_pred"]
  if len(distances_gt_to_pred) > 0:  # pylint: disable=g-explicit-length-test
    surfel_areas_cum_gt = np.cumsum(surfel_areas_gt) / np.sum(surfel_areas_gt)
    idx = np.searchsorted(surfel_areas_cum_gt, percent/100.0)
    perc_distance_gt_to_pred = distances_gt_to_pred[
        min(idx, len(distances_gt_to_pred)-1)]
  else:
    perc_distance_gt_to_pred = np.inf

  if len(distances_pred_to_gt) > 0:  # pylint: disable=g-explicit-length-test
    surfel_areas_cum_pred = (np.cumsum(surfel_areas_pred) /
                             np.sum(surfel_areas_pred))
    idx = np.searchsorted(surfel_areas_cum_pred, percent/100.0)
    perc_distance_pred_to_gt = distances_pred_to_gt[
        min(idx, len(distances_pred_to_gt)-1)]
  else:
    perc_distance_pred_to_gt = np.inf

  return max(perc_distance_gt_to_pred, perc_distance_pred_to_gt)

class DiceScore(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.Dice = 0 

    def update(self, preds: Tensor, target: Tensor) -> None:
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        shape = preds.shape
        Dice_all_im = 0
        self.Dice = 0
        if len(shape) > 3:
            for num in range(shape[0]):
                single_pred = preds[num, 0, :, :]*1
                single_target = target[num, 0, :, :]
                Dice_all_im += 2*torch.sum(torch.mul(single_pred, single_target)) / (torch.sum(single_target) + torch.sum(single_pred))
            self.Dice = Dice_all_im/shape[0]
        else:
            self.Dice += 2*torch.sum(torch.mul(preds, target)) / (torch.sum(target) + torch.sum(preds))
        
    def compute(self) -> Tensor:
        return self.Dice

class JacScore(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.Jac = 0

    def update(self, preds: Tensor, target: Tensor) -> None:
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        shape = preds.shape
        Jac_all_im = 0
        self.Jac = 0
        if len(shape) > 3:
            for num in range(shape[0]):
                single_pred = preds[num, 0, :, :]*1
                single_target = target[num, 0, :, :]
                Jac_all_im += torch.sum(torch.mul(single_pred, single_target)) / (torch.sum(single_target) + torch.sum(single_pred) - torch.sum(torch.mul(single_pred, single_target)))
            self.Jac = Jac_all_im/shape[0]
        else: 
            self.Jac += torch.sum(torch.mul(preds, target)) / (torch.sum(target) + torch.sum(preds) - torch.sum(torch.mul(preds, target)))
        
    def compute(self) -> Tensor:
        return self.Jac

class PixelAcc(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.pixel_ACC = 0 

    def update(self, preds: Tensor, target: Tensor) -> None:
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        shape = preds.shape
        self.pixel_ACC = 0
        pixel_ACC_all_ims = 0
        if len(shape) > 3:
            for num in range(shape[0]):
                single_pred = preds[num, 0, :, :]
                single_target = target[num, 0, :, :]
                pixel_ACC_all_ims += torch.sum(torch.mul(single_pred, single_target)) / (torch.sum(single_target) + torch.sum(single_pred) - torch.sum(torch.mul(single_pred, single_target)))
            self.pixel_ACC = pixel_ACC_all_ims/shape[0]
        else:
            self.pixel_ACC += torch.sum(torch.mul(preds, target)) / (torch.sum(target) + torch.sum(preds) - torch.sum(torch.mul(preds, target)))

    def compute(self) -> Tensor:
        return self.pixel_ACC

class PixelPrecision(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.pixel_Prec = 0

    def update(self, preds: Tensor, target: Tensor) -> None:
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        shape = preds.shape
        self.pixel_Prec = 0
        pixel_Prec_all_ims = 0
        if len(shape) > 3:
            for num in range(shape[0]):
                single_pred = preds[num, 0, :, :]
                single_target = target[num, 0, :, :]

                pixel_Prec_all_ims += torch.mul(single_pred, single_target).sum() / single_pred.sum()
            self.pixel_Prec = pixel_Prec_all_ims/shape[0]
        else: 
            self.pixel_Prec += torch.mul(preds, target).sum() / preds.sum()

    def compute(self) -> Tensor:
        return self.pixel_Prec

class PixelRecall(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.pixel_Rec = 0

    def update(self, preds: Tensor, target: Tensor) -> None:
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        shape = preds.shape
        self.pixel_Rec = 0
        pixel_Rec_all_ims = 0
        if len(shape) >= 3:
            for num in range(shape[0]):
                single_pred = preds[num, 0, :, :]
                single_target = target[num, 0, :, :]

                pixel_Rec_all_ims += torch.mul(single_pred, single_target).sum() / single_target.sum()
            self.pixel_Rec = pixel_Rec_all_ims/shape[0]
        else: 
            self.pixel_Prec += torch.mul(preds, target).sum() / target.sum() # * shape[1]*shape[2]

    def compute(self) -> Tensor:
        return self.pixel_Rec

def get_Dice_Jaccard_array(image_tensor_input, y_tensor_input):

    Jaccard_Index = JacScore() #JaccardIndex(task="multiclass", num_classes=2)
    Dice_metric = DiceScore()

    dice_jaccard_tensor = torch.Tensor([0, 0]).reshape(1,2)
    for im_num in range(image_tensor_input.shape[0]):
        dice_jaccard_tensor = torch.cat((dice_jaccard_tensor, torch.Tensor([Dice_metric(image_tensor_input[im_num,:,:,:], y_tensor_input[im_num,:,:,:]), Jaccard_Index(image_tensor_input[im_num,:,:,:], y_tensor_input[im_num,:,:,:])]).reshape(1,2)), 0)
    return dice_jaccard_tensor[1:,:]

class HD95(Metric):
    def __init__(self, percent, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.percent = percent
        self.HD95 = 0 

    def update(self, preds: Tensor, target: Tensor) -> None:
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")
        
        shape = preds.shape
        self.HD95 = 0
        HD95_all_ims = 0

        for num in range(shape[0]):
            surface_dists = compute_surface_distances(target[num, 0, :,:].detach().cpu().to(torch.bool).numpy(), preds[num, 0, :,:].detach().cpu().to(torch.bool).numpy(), [1, 1])
            HD95_all_ims += compute_robust_hausdorff(surface_dists, self.percent)
        self.HD95 = HD95_all_ims/shape[0]
    
    def compute(self) -> Tensor:
        return self.HD95

class LitModel_SimpleSemanticSeg(pl.LightningModule):
    def __init__(self, model, batch_size, learning_rate, model_name, save_im_fp, loss_fn): #, lamda): #optimizer): 
        super().__init__()
        self.model = model
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(
                                params=self.model.parameters(), 
                                lr=self.learning_rate, maximize=False)
        self.save_path = model_name
        self.image_path = save_im_fp
        self.loss_fn = loss_fn
        self.PCC = PearsonCorrCoef().to(device='cuda')
        self.Jaccard_Index = JacScore().to(device='cuda') #task="multiclass", num_classes=2).to(device='cpu')
        self.Dice_metric = DiceScore().to(device='cuda')
        self.pixel_acc = PixelAcc()
        self.pixel_pre = PixelPrecision()
        self.pixel_rec = PixelRecall()
        self.HD = HausdorffDistance(num_classes=2).to(device='cuda')
        self.HD_95 = HD95(95)
        self.dice_jaccard_tensor = torch.tensor([]).cuda()

    def training_step(self, batch, batch_idx):
        dif_area_out = 0
        x, y, _ = batch
        bground = F.sigmoid(self.model(x.float()))
        
        image_out = torch.where(bground.float() >= 0.5, 1 , 0)
        image_out = image_out > 0.5
        y_out = torch.where(y >= 0.5, 1 , 0)
        
        if y_out.shape[2] != image_out.shape[2] or y_out.shape[3] != image_out.shape[3]:
            transform_resize = torchvision.transforms.Resize((y_out.shape[2],  y_out.shape[3]), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
            image_out = transform_resize(image_out)
            bground = transform_resize(bground)

        loss =+ self.loss_fn(image_out.cuda().float(), y_out.float())
        dif_area_out = area_calculation(y_out)*0.00017936 - area_calculation(image_out)*0.00017936 + dif_area_out

        self.log("train_loss", np.nan_to_num(loss.detach().cpu().numpy()), on_step = False, on_epoch = True)
        self.log("Train Dice Metric: ", np.nan_to_num(self.Dice_metric(image_out.cpu(), y_out.cpu())), on_step=False, on_epoch=True)
        self.log("Train Pixel Acc: ", np.nan_to_num(self.pixel_acc(image_out.cpu(), y_out.cpu())), on_step=False, on_epoch=True)
        self.log("Train Pixel Precision: ", np.nan_to_num(self.pixel_pre(image_out.cpu(), y_out.cpu())), on_step=False, on_epoch=True)
        self.log("Train Pixel Recall: ", np.nan_to_num(self.pixel_rec(image_out.cpu(), y_out.cpu())), on_step=False, on_epoch=True)
        self.log("Train HD: ", torch.nan_to_num(self.HD(image_out.cuda(), y_out.cuda())), on_step=False, on_epoch=True)
        self.log("Train Jaccard Index: ", np.nan_to_num(self.Jaccard_Index(image_out.cpu(), y_out.cpu())), on_step=False, on_epoch=True)
        self.log("Train Area Difference: ", np.nan_to_num(dif_area_out.mean().detach().cpu().numpy()), on_step=False, on_epoch=True)
        self.log("Train Area Standard Deviation : ", np.nan_to_num(dif_area_out.std().detach().cpu().numpy()), on_step=False, on_epoch=True)
        return self.loss_fn(bground.cuda().float(), y_out.float())

    def validation_step(self, batch, batch_idx):
        x, y, batch_index = batch
        bground = self.model(x.float())
        image_out = torch.where(bground.float() >= 0.5, 1 , 0)
        image_out = image_out > 0.5
        y_out = torch.where(y >= 0.5, 1 , 0)
        
        transform = T.ToPILImage()

        if y_out.shape[2] != image_out.shape[2] or y_out.shape[3] != image_out.shape[3]:
            transform_resize = torchvision.transforms.Resize((y_out.shape[2],  y_out.shape[3]), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
            image_out = transform_resize(image_out)

        val_loss =+ self.loss_fn(image_out.cuda().float(), y_out.cuda().float()) #+ self.lamda*metrics.hausdorff_distance(bground.cpu().bool(), y_out.cpu().bool())
        for num in range(image_out.shape[0]):
            transform(torch.tensor(image_out[num, :,:, :]*255, dtype=torch.uint8)).save(self.image_path + "val_batch_bg_faz_seg_" + str(num) + "_sonia_" + str(self.learning_rate) + "_" + str(self.batch_size) + "_batch_ind_" + batch_index[num] + ".tiff", quality=100)
            
        dif_area_out = area_calculation(y_out)*0.00017936 - area_calculation(image_out)*0.00017936

        self.log("val_loss",  np.nan_to_num(val_loss.detach().cpu().numpy()), on_step = False, on_epoch = True)
        self.log("Val Dice Metric: ",  np.nan_to_num(self.Dice_metric(image_out.cpu(), y_out.cpu())), on_step=False, on_epoch=True)
        self.log("Val Pixel Acc: ", np.nan_to_num(self.pixel_acc(image_out.cpu(), y_out.cpu())), on_step=False, on_epoch=True)
        self.log("Val Pixel Precision: ", np.nan_to_num(self.pixel_pre(image_out.cpu(), y_out.cpu())), on_step=False, on_epoch=True)
        self.log("Val Pixel Recall: ", np.nan_to_num(self.pixel_rec(image_out.cpu(), y_out.cpu())), on_step=False, on_epoch=True)
        self.log("Val HD: ", torch.nan_to_num(self.HD(image_out.cuda(), y_out.cuda())), on_step=False, on_epoch=True)
        self.log("Val Jaccard Index: ",  np.nan_to_num(self.Jaccard_Index(image_out.detach().cpu(), y_out.detach().cpu())), on_step=False, on_epoch=True)
        self.log("Val Area Difference: ",  np.nan_to_num(dif_area_out.mean().detach().cpu().numpy()), on_step=False, on_epoch=True)
        self.log("Val Area Standard Deviation: ",  np.nan_to_num(dif_area_out.std().detach().cpu().numpy()), on_step=False, on_epoch=True)
        self.log("Val PCC Area: ",  torch.nan_to_num(self.PCC(area_calculation(y_out).cuda()*0.00017936, area_calculation(image_out).cuda()*0.00017936)), on_step=False, on_epoch=True)
        self.log("Val PCC Perimiter: ",  torch.nan_to_num(self.PCC(perimiter_calculation(y_out.cuda()).cuda()*0.01339, perimiter_calculation(image_out.cuda()).cuda()*0.01339)), on_step=False, on_epoch=True)
        self.log("Val PCC Roundness: ", self.PCC(roundness_calculation(y_out).cuda(), roundness_calculation(image_out).cuda()), on_step=False, on_epoch=True)
        return self.loss_fn(image_out.cuda().float(), y_out.cuda().float())

    def test_step(self, batch, batch_idx): 
        x, y, batch_index = batch 
        bground = self.model(x.float())
        image_out = torch.where(bground.float() >= 0.5, 1 , 0)
        image_out = image_out > 0.5
        y_out = torch.where(y >= 0.5, 1 , 0)

        if y_out.shape[2] != image_out.shape[2] or y_out.shape[3] != image_out.shape[3]:
            transform_resize = torchvision.transforms.Resize((y_out.shape[2],  y_out.shape[3]), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
            image_out = transform_resize(image_out)

        image_out_clean_list = torch.ones((1, 1, image_out.shape[2], image_out.shape[3]))
        for num in range(image_out.shape[0]):
            image_out_clean = torch.tensor(get_LargsetContArea(torch.squeeze(image_out[num, :, :, :].cpu()).numpy(), image_out.shape[2], image_out.shape[3])).reshape(1, 1, image_out.shape[2], image_out.shape[3])
            image_out_clean_list = torch.cat((image_out_clean_list, image_out_clean), dim = 0)
        image_out_clean_list = image_out_clean_list[1:]  
        image_out_clean_list = image_out_clean_list > 0.5
        save_image_batch_3chan(image_out_clean_list, x, self.image_path + "test_batch_bg", y_out.cpu(), self.learning_rate, self.batch_size)
        transform = T.ToPILImage()

        for num in range(image_out_clean_list.shape[0]):
            img_label = batch_index[num]
            transform(torch.tensor(image_out_clean_list[num, :,:, :]*255, dtype=torch.uint8)).save(self.image_path + "test_batch_bg_faz_seg_" + str(num) + "_sonia_" + str(self.learning_rate) + "_" + str(self.batch_size) + "_batch_ind_" + img_label + ".tiff", quality=100)
            
        self.dice_jaccard_tensor = get_Dice_Jaccard_array(image_out_clean_list.cpu(), y_out.cpu())

        df = pd.DataFrame(self.dice_jaccard_tensor.detach().cpu(), columns=['Dice Score', 'Jaccard Score'])
        df.to_csv("Dice_Jaccard_Index_array/entire_dataset_results.csv")

        m1 = df.mean(axis=0)
        st1 = df.std(axis=0)
        plt.close('all')
        fig, ax = plt.subplots()
        bp = ax.boxplot([np.array((torch.transpose(self.dice_jaccard_tensor.detach().cpu(), 1, 0) * 100)[0]), np.array((torch.transpose(self.dice_jaccard_tensor.detach().cpu(), 1, 0) * 100)[1])], meanline=True, showmeans= True)  # , labels=["Dice Metric", "Jaccard Index"]

        for i, line in enumerate(bp['medians']):
            x, y = line.get_xydata()[1]
            text = ' μ={:.4f}\n σ={:.4f}'.format(m1[i], st1[i])
            ax.annotate(text, xy=(x, y))
        ax.set_xticks([1, 2], labels=["Dice Metric", "Jaccard Index"])
        fig.savefig("model_ims/box_plot_lr_" + str(self.learning_rate) + "_all_images_.png") #home/vm/model_ims/
        fig.clf()

        if y_out.shape[2] != image_out.shape[2] or y_out.shape[3] != image_out.shape[3]:
            transform_resize = torchvision.transforms.Resize((image_out.shape[2],  image_out.shape[3]), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
            y_out = transform_resize(y_out)

        test_loss =+ self.loss_fn(image_out.float(), y_out.float())
        dif_area_out = area_calculation(y_out)*0.00017936 - area_calculation(image_out)*0.00017936
        dif_peri_out = perimiter_calculation(y_out)*0.01339 - perimiter_calculation(image_out)*0.01339
        plt.close('all')
        fig, ax = plt.subplots(figsize = (8,5))
        pplot = sm.graphics.mean_diff_plot(area_calculation(y_out).cpu().numpy()*0.00017936, area_calculation(image_out).cpu().numpy()*0.00017936, ax=ax)
        fig.savefig("model_ims/bland_altman_area_lr_" + str(self.learning_rate) + "_all_images_.png") # /home/vm/model_ims/
        fig.clf()
        
        plt.close('all')
        fig, ax = plt.subplots(1, figsize = (8,5))
        pplot = sm.graphics.mean_diff_plot(perimiter_calculation(y_out).cpu().numpy()*0.01339, perimiter_calculation(image_out).cpu().numpy()*0.01339, ax=ax)
        fig.savefig("model_ims/bland_altman_perimiter_lr_" + str(self.learning_rate) + "_all_images_.png") # /home/vm/model_ims/
        fig.clf()

        self.log("test_loss",  np.nan_to_num(test_loss.detach().cpu().numpy()), on_step = False, on_epoch = True)
        self.log("Test Jaccard Index: ",  np.nan_to_num(self.Jaccard_Index(image_out_clean_list.cpu(), y_out.cpu())), on_step=False, on_epoch=True)
        self.log("Test Dice Metric: ",  np.nan_to_num(self.Dice_metric(image_out_clean_list.cpu(), y_out.cpu())), on_step=False, on_epoch=True)
        self.log("Test Pixel Acc: ", np.nan_to_num(self.pixel_acc(image_out_clean_list.cpu(), y_out.cpu())), on_step=False, on_epoch=True)
        self.log("Test Pixel Precision: ", np.nan_to_num(self.pixel_pre(image_out_clean_list.cpu(), y_out.cpu())), on_step=False, on_epoch=True)
        self.log("Test Pixel Recall: ", np.nan_to_num(self.pixel_rec(image_out_clean_list.cpu(), y_out.cpu())), on_step=False, on_epoch=True)
        self.log("Test HD: ", torch.nan_to_num(self.HD(image_out_clean_list.cpu(), y_out.cpu())), on_step=False, on_epoch=True)
        self.log("Test HD 95: ", np.nan_to_num(self.HD_95(image_out_clean_list.cpu(), y_out.cpu())), on_step=False, on_epoch=True)
        self.log("Test Area Difference: ",  np.nan_to_num(dif_area_out.mean().detach().cpu().numpy()), on_step=False, on_epoch=True)
        self.log("Test Perimeter Difference: ",  np.nan_to_num(dif_peri_out.mean().detach().cpu().numpy()), on_step=False, on_epoch=True)
        self.log("Test PCC Area: ",  torch.nan_to_num(self.PCC(area_calculation(y_out).cuda()*0.00017936, area_calculation(image_out_clean_list).cuda()*0.00017936)), on_step=False, on_epoch=True)
        self.log("Test PCC Perimiter: ",  torch.nan_to_num(self.PCC(perimiter_calculation(y_out.cuda()).cuda()*0.01339, perimiter_calculation(image_out_clean_list.cuda()).cuda()*0.01339)), on_step=False, on_epoch=True)
        self.log("Test PCC Roundness: ", torch.nan_to_num(self.PCC(roundness_calculation(y_out).cuda(), roundness_calculation(image_out_clean_list).cuda())), on_step=False, on_epoch=True)

        return test_loss
    
    def forward(self, x, num):
        bground = self.model(x.float())
        image_out = torch.where(bground.float() >= 0.5, 1 , 0)
        image_out = image_out > 0.5
        image_out_clean_list = torch.ones((1, 1, image_out.shape[2], image_out.shape[3]))
        transform = T.ToPILImage()
        transform(torch.tensor(image_out_clean_list*255, dtype=torch.uint8).squeeze(0)).save(self.image_path + "test_batch_bg_faz_seg_" + str(num) + "_sonia_" + str(self.learning_rate) + "_" + str(self.batch_size) + "_batch_ind_" + str(num) + ".tiff", quality=100)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
                                params=self.model.parameters(), 
                                lr=self.learning_rate)
        return optimizer


class LitModel_SimpleSemanticSegOphtal2(pl.LightningModule):
    def __init__(self, model, batch_size, learning_rate, model_name, save_im_fp, loss_fn):  # , lamda): #optimizer):
        super().__init__()
        self.model = model

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.learning_rate, maximize=False)
        self.save_path = model_name
        self.image_path = save_im_fp
        self.loss_fn = loss_fn
        self.PCC = PearsonCorrCoef().to(device='cuda')
        self.Jaccard_Index = JacScore().to(device='cuda')  # task="multiclass", num_classes=2).to(device='cpu')
        self.Dice_metric = DiceScore().to(device='cuda')
        self.pixel_acc = PixelAcc()
        self.pixel_pre = PixelPrecision()
        self.pixel_rec = PixelRecall()
        self.HD = HausdorffDistance(num_classes=2).to(device='cuda')
        self.HD_95 = HD95(95)
        self.dice_jaccard_tensor = torch.tensor([]).cuda()

    def training_step(self, batch, batch_idx):
        dif_area_out = 0
        x, y, _, _ = batch
        bground = F.sigmoid(self.model(x.float()))

        image_out = torch.where(bground.float() >= 0.5, 1, 0)
        image_out = image_out > 0.5
        y_out = torch.where(y >= 0.5, 1, 0)

        if y_out.shape[2] != image_out.shape[2] or y_out.shape[3] != image_out.shape[3]:
            transform_resize = torchvision.transforms.Resize((y_out.shape[2], y_out.shape[3]),
                                                             interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
            image_out = transform_resize(image_out)
            bground = transform_resize(bground)

        loss = + self.loss_fn(image_out.cuda().float(), y_out.float())
        dif_area_out = area_calculation(y_out) * 0.00017936 - area_calculation(image_out) * 0.00017936 + dif_area_out

        self.log("train_loss", np.nan_to_num(loss.detach().cpu().numpy()), on_step=False, on_epoch=True)
        self.log("Train Dice Metric: ", np.nan_to_num(self.Dice_metric(image_out.cpu(), y_out.cpu())), on_step=False,
                 on_epoch=True)
        self.log("Train Pixel Acc: ", np.nan_to_num(self.pixel_acc(image_out.cpu(), y_out.cpu())), on_step=False,
                 on_epoch=True)
        self.log("Train Pixel Precision: ", np.nan_to_num(self.pixel_pre(image_out.cpu(), y_out.cpu())), on_step=False,
                 on_epoch=True)
        self.log("Train Pixel Recall: ", np.nan_to_num(self.pixel_rec(image_out.cpu(), y_out.cpu())), on_step=False,
                 on_epoch=True)
        self.log("Train HD: ", torch.nan_to_num(self.HD(image_out.cpu(), y_out.cpu())), on_step=False, on_epoch=True)
        self.log("Train Jaccard Index: ", np.nan_to_num(self.Jaccard_Index(image_out.cpu(), y_out.cpu())),
                 on_step=False, on_epoch=True)
        self.log("Train Area Difference: ", np.nan_to_num(dif_area_out.mean().detach().cpu().numpy()), on_step=False,
                 on_epoch=True)
        self.log("Train Area Standard Deviation : ", np.nan_to_num(dif_area_out.std().detach().cpu().numpy()),
                 on_step=False, on_epoch=True)
        return self.loss_fn(bground.cuda().float(), y_out.float())

    def validation_step(self, batch, batch_idx):
        x, y, _, _ = batch
        bground = self.model(x.float())
        image_out = torch.where(bground.float() >= 0.5, 1, 0)
        image_out = image_out > 0.5
        y_out = torch.where(y >= 0.5, 1, 0)

        transform = T.ToPILImage()

        if y_out.shape[2] != image_out.shape[2] or y_out.shape[3] != image_out.shape[3]:
            transform_resize = torchvision.transforms.Resize((y_out.shape[2], y_out.shape[3]),
                                                             interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
            image_out = transform_resize(image_out)

        val_loss = + self.loss_fn(image_out.cuda().float(),
                                  y_out.cuda().float())  # + self.lamda*metrics.hausdorff_distance(bground.cpu().bool(), y_out.cpu().bool())
        for num in range(image_out.shape[0]):
            transform(torch.tensor(image_out[num, :, :, :] * 255, dtype=torch.uint8)).save(
                self.image_path + "val_batch_bg_faz_seg_" + str(num) + "_sonia_" + str(self.learning_rate) + "_" + str(
                    self.batch_size) + ".tiff", quality=100)

        dif_area_out = area_calculation(y_out) * 0.00017936 - area_calculation(image_out) * 0.00017936

        self.log("val_loss", np.nan_to_num(val_loss.detach().cpu().numpy()), on_step=False, on_epoch=True)
        self.log("Val Dice Metric: ", np.nan_to_num(self.Dice_metric(image_out.cpu(), y_out.cpu())), on_step=False,
                 on_epoch=True)
        self.log("Val Pixel Acc: ", np.nan_to_num(self.pixel_acc(image_out.cpu(), y_out.cpu())), on_step=False,
                 on_epoch=True)
        self.log("Val Pixel Precision: ", np.nan_to_num(self.pixel_pre(image_out.cpu(), y_out.cpu())), on_step=False,
                 on_epoch=True)
        self.log("Val Pixel Recall: ", np.nan_to_num(self.pixel_rec(image_out.cpu(), y_out.cpu())), on_step=False,
                 on_epoch=True)
        self.log("Val HD: ", torch.nan_to_num(self.HD(image_out.cuda(), y_out.cuda())), on_step=False, on_epoch=True)
        self.log("Val Jaccard Index: ",
                 np.nan_to_num(self.Jaccard_Index(image_out.detach().cpu(), y_out.detach().cpu())), on_step=False,
                 on_epoch=True)
        self.log("Val Area Difference: ", np.nan_to_num(dif_area_out.mean().detach().cpu().numpy()), on_step=False,
                 on_epoch=True)
        self.log("Val Area Standard Deviation: ", np.nan_to_num(dif_area_out.std().detach().cpu().numpy()),
                 on_step=False, on_epoch=True)
        self.log("Val PCC Area: ", torch.nan_to_num(
            self.PCC(area_calculation(y_out).cuda() * 0.00017936, area_calculation(image_out).cuda() * 0.00017936)),
                 on_step=False, on_epoch=True)
        self.log("Val PCC Perimiter: ", torch.nan_to_num(self.PCC(perimiter_calculation(y_out.cuda()).cuda() * 0.01339,
                                                                  perimiter_calculation(
                                                                      image_out.cuda()).cuda() * 0.01339)),
                 on_step=False, on_epoch=True)
        self.log("Val PCC Roundness: ",
                 self.PCC(roundness_calculation(y_out).cuda(), roundness_calculation(image_out).cuda()), on_step=False,
                 on_epoch=True)
        return self.loss_fn(image_out.cuda().float(), y_out.cuda().float())

    def test_step(self, batch, batch_idx):
        x, y, y_ophtal_2, _ = batch
        bground = self.model(x.float())
        image_out = torch.where(bground.float() >= 0.5, 1, 0)
        image_out = image_out > 0.5
        y_out = torch.where(y >= 0.5, 1, 0)
        y_ophtal_2 = torch.where(y_ophtal_2 >= 0.5, 1, 0)

        # Ophtalomologist 1
        if y_out.shape[2] != image_out.shape[2] or y_out.shape[3] != image_out.shape[3]:
            transform_resize = torchvision.transforms.Resize((y_out.shape[2], y_out.shape[3]),
                                                             interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
            image_out = transform_resize(image_out)

        image_out_clean_list = torch.ones((1, 1, image_out.shape[2], image_out.shape[3]))
        for num in range(image_out.shape[0]):
            image_out_clean = torch.tensor(
                get_LargsetContArea(torch.squeeze(image_out[num, :, :, :].cpu()).numpy(), image_out.shape[2],
                                    image_out.shape[3])).reshape(1, 1, image_out.shape[2], image_out.shape[3])
            image_out_clean_list = torch.cat((image_out_clean_list, image_out_clean), dim=0)
        image_out_clean_list = image_out_clean_list[1:]
        image_out_clean_list = image_out_clean_list > 0.5
        
        save_image_batch_3chan(image_out_clean_list, x, self.image_path + "test_batch_bg", y_out.cpu(),
                               self.learning_rate, self.batch_size)
        transform = T.ToPILImage()

        for num in range(image_out_clean_list.shape[0]):
            transform(torch.tensor(image_out_clean_list[num, :, :, :] * 255, dtype=torch.uint8)).save(
                self.image_path + "test_batch_bg_faz_seg_" + str(num) + "_sonia_" + str(
                    self.learning_rate) + "_" + str(self.batch_size) + ".tiff", quality=100)

        self.dice_jaccard_tensor = get_Dice_Jaccard_array(image_out_clean_list.cpu(), y_out.cpu())

        df = pd.DataFrame(self.dice_jaccard_tensor.detach().cpu(), columns=['Dice Score', 'Jaccard Score'])
        df.to_csv("Dice_Jaccard_Index_array/entire_dataset_results.csv")

        m1 = df.mean(axis=0)
        st1 = df.std(axis=0)
        plt.close('all')
        fig, ax = plt.subplots()
        bp = ax.boxplot([np.array((torch.transpose(self.dice_jaccard_tensor.detach().cpu(), 1, 0) * 100)[0]),
                         np.array((torch.transpose(self.dice_jaccard_tensor.detach().cpu(), 1, 0) * 100)[1])],
                        meanline=True, showmeans=True)

        for i, line in enumerate(bp['medians']):
            x_1, y = line.get_xydata()[1]
            text = ' μ={:.4f}\n σ={:.4f}'.format(m1[i], st1[i])
            ax.annotate(text, xy=(x_1, y))
        ax.set_xticks([1, 2], labels=["Dice Metric", "Jaccard Index"])
        fig.savefig("model_ims/box_plot_lr_" + str(self.learning_rate) + "_all_images_.png")
        fig.clf()

        if y_out.shape[2] != image_out.shape[2] or y_out.shape[3] != image_out.shape[3]:
            transform_resize = torchvision.transforms.Resize((image_out.shape[2], image_out.shape[3]),
                                                             interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
            y_out = transform_resize(y_out)

        test_loss = + self.loss_fn(image_out.float(), y_out.float())
        dif_area_out = area_calculation(y_out) * 0.00017936 - area_calculation(image_out) * 0.00017936
        dif_peri_out = perimiter_calculation(y_out) * 0.01339 - perimiter_calculation(image_out) * 0.01339

        plt.close('all')
        fig, ax = plt.subplots(figsize=(8, 5))
        pplot = sm.graphics.mean_diff_plot(area_calculation(y_out).cpu().numpy() * 0.00017936,
                                           area_calculation(image_out).cpu().numpy() * 0.00017936, ax=ax)
        fig.savefig("model_ims/bland_altman_area_lr_" + str(
            self.learning_rate) + "_all_images_.png")  # /home/vm/model_ims/
        fig.clf()

        plt.close('all')
        fig, ax = plt.subplots(1, figsize=(8, 5))
        pplot = sm.graphics.mean_diff_plot(perimiter_calculation(y_out).cpu().numpy() * 0.01339,
                                           perimiter_calculation(image_out).cpu().numpy() * 0.01339, ax=ax)
        fig.savefig("model_ims/bland_altman_perimiter_lr_" + str(
            self.learning_rate) + "_all_images_.png")  # /home/vm/model_ims/
        fig.clf()

        # Ophtalomologist 2
        save_image_batch_3chan(image_out_clean_list, x, self.image_path + "test_batch_bg_", y_ophtal_2.cpu(),
                               self.learning_rate, str(self.batch_size) + "_ophtal_2")
        transform = T.ToPILImage()

        for num in range(image_out_clean_list.shape[0]):
            transform(torch.tensor(image_out_clean_list[num, :, :, :] * 255, dtype=torch.uint8)).save(
                self.image_path + "test_batch_bg_faz_seg_" + str(num) + "_sonia_" + str(
                    self.learning_rate) + "_" + str(self.batch_size) + "_y_ophtal_2.tiff", quality=100)

        self.dice_jaccard_tensor_ophtal_2 = get_Dice_Jaccard_array(image_out_clean_list.cpu(), y_ophtal_2.cpu())

        df = pd.DataFrame(self.dice_jaccard_tensor_ophtal_2.detach().cpu(), columns=['Dice Score', 'Jaccard Score'])
        df.to_csv("Dice_Jaccard_Index_array/entire_dataset_results_ophtal_2.csv")

        m1 = df.mean(axis=0)
        st1 = df.std(axis=0)
        plt.close('all')
        fig, ax = plt.subplots()
        bp = ax.boxplot([np.array((torch.transpose(self.dice_jaccard_tensor_ophtal_2.detach().cpu(), 1, 0) * 100)[0]),
                         np.array((torch.transpose(self.dice_jaccard_tensor_ophtal_2.detach().cpu(), 1, 0) * 100)[1])],
                        meanline=True, showmeans=True)  # , labels=["Dice Metric", "Jaccard Index"]

        for i, line in enumerate(bp['medians']):
            x_2, y = line.get_xydata()[1]
            text = ' μ={:.4f}\n σ={:.4f}'.format(m1[i], st1[i])
            ax.annotate(text, xy=(x_2, y))
        ax.set_xticks([1, 2], labels=["Dice Metric", "Jaccard Index"])
        fig.savefig("model_ims/box_plot_lr_" + str(
            self.learning_rate) + "_all_images_ophtal_2.png")  # home/vm/model_ims/
        fig.clf()

        if y_ophtal_2.shape[2] != image_out.shape[2] or y_ophtal_2.shape[3] != image_out.shape[3]:
            transform_resize = torchvision.transforms.Resize((image_out.shape[2], image_out.shape[3]),
                                                             interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
            y_ophtal_2 = transform_resize(y_ophtal_2)

        test_loss = + self.loss_fn(image_out.float(), y_ophtal_2.float())
        dif_area_out_ophtal2 = area_calculation(y_ophtal_2) * 0.00017936 - area_calculation(image_out) * 0.00017936
        dif_peri_out_ophtal2 = perimiter_calculation(y_ophtal_2) * 0.01339 - perimiter_calculation(image_out) * 0.01339

        plt.close('all')
        fig, ax = plt.subplots(figsize=(8, 5))
        pplot = sm.graphics.mean_diff_plot(area_calculation(y_ophtal_2).cpu().numpy() * 0.00017936,
                                           area_calculation(image_out).cpu().numpy() * 0.00017936, ax=ax)
        fig.savefig("model_ims/bland_altman_area_lr_" + str(
            self.learning_rate) + "_all_images_ophtal_2.png")  # /home/vm/model_ims/
        fig.clf()

        plt.close('all')
        fig, ax = plt.subplots(1, figsize=(8, 5))
        pplot = sm.graphics.mean_diff_plot(perimiter_calculation(y_ophtal_2).cpu().numpy() * 0.01339,
                                           perimiter_calculation(image_out).cpu().numpy() * 0.01339, ax=ax)
        fig.savefig("model_ims/bland_altman_perimiter_lr_" + str(
            self.learning_rate) + "_all_images_ophtal_2.png")  # /home/vm/model_ims/
        fig.clf()

        # Ophtalomologist 1
        self.log("test_loss", np.nan_to_num(test_loss.detach().cpu().numpy()), on_step=False, on_epoch=True)
        self.log("Test Jaccard Index: ", np.nan_to_num(self.Jaccard_Index(image_out_clean_list.cpu(), y_out.cpu())),
                 on_step=False, on_epoch=True)
        self.log("Test Dice Metric: ", np.nan_to_num(self.Dice_metric(image_out_clean_list.cpu(), y_out.cpu())),
                 on_step=False, on_epoch=True)
        self.log("Test Pixel Acc: ", np.nan_to_num(self.pixel_acc(image_out_clean_list.cpu(), y_out.cpu())),
                 on_step=False, on_epoch=True)
        self.log("Test Pixel Precision: ", np.nan_to_num(self.pixel_pre(image_out_clean_list.cpu(), y_out.cpu())),
                 on_step=False, on_epoch=True)
        self.log("Test Pixel Recall: ", np.nan_to_num(self.pixel_rec(image_out_clean_list.cpu(), y_out.cpu())),
                 on_step=False, on_epoch=True)
        self.log("Test HD: ", torch.nan_to_num(self.HD(image_out_clean_list.cpu(), y_out.cpu())), on_step=False,
                 on_epoch=True)
        self.log("Test HD 95: ", np.nan_to_num(self.HD_95(image_out_clean_list.cpu(), y_out.cpu())), on_step=False,
                 on_epoch=True)
        self.log("Test Area Difference: ", np.nan_to_num(dif_area_out.mean().detach().cpu().numpy()), on_step=False,
                 on_epoch=True)
        self.log("Test Perimeter Difference: ", np.nan_to_num(dif_peri_out.mean().detach().cpu().numpy()),
                 on_step=False, on_epoch=True)
        self.log("Test PCC Area: ", torch.nan_to_num(self.PCC(area_calculation(y_out).cuda() * 0.00017936,
                                                              area_calculation(
                                                                  image_out_clean_list).cuda() * 0.00017936)),
                 on_step=False, on_epoch=True)
        self.log("Test PCC Perimiter: ", torch.nan_to_num(self.PCC(perimiter_calculation(y_out.cuda()).cuda() * 0.01339,
                                                                   perimiter_calculation(
                                                                       image_out_clean_list.cuda()).cuda() * 0.01339)),
                 on_step=False, on_epoch=True)
        self.log("Test PCC Roundness: ", torch.nan_to_num(
            self.PCC(roundness_calculation(y_out).cuda(), roundness_calculation(image_out_clean_list).cuda())),
                 on_step=False, on_epoch=True)

        # Ophtalomologist 2
        self.log("Test Jaccard Index Ophtal 2: ", np.nan_to_num(self.Jaccard_Index(image_out_clean_list.cpu(), y_ophtal_2.cpu())),
                 on_step=False, on_epoch=True)
        self.log("Test Dice Metric Ophtal 2: ", np.nan_to_num(self.Dice_metric(image_out_clean_list.cpu(), y_ophtal_2.cpu())),
                 on_step=False, on_epoch=True)
        self.log("Test Pixel Acc Ophtal 2: ", np.nan_to_num(self.pixel_acc(image_out_clean_list.cpu(), y_ophtal_2.cpu())),
                 on_step=False, on_epoch=True)
        self.log("Test Pixel Precision Ophtal 2: ", np.nan_to_num(self.pixel_pre(image_out_clean_list.cpu(), y_ophtal_2.cpu())),
                 on_step=False, on_epoch=True)
        self.log("Test Pixel Recall Ophtal 2: ", np.nan_to_num(self.pixel_rec(image_out_clean_list.cpu(), y_ophtal_2.cpu())),
                 on_step=False, on_epoch=True)
        self.log("Test HD Ophtal 2: ", torch.nan_to_num(self.HD(image_out_clean_list.cpu(), y_ophtal_2.cpu())), on_step=False,
                 on_epoch=True)
        self.log("Test HD 95 Ophtal 2: ", np.nan_to_num(self.HD_95(image_out_clean_list.cpu(), y_ophtal_2.cpu())), on_step=False,
                 on_epoch=True)
        self.log("Test Area Difference Ophtal 2: ", np.nan_to_num(dif_area_out_ophtal2.mean().detach().cpu().numpy()), on_step=False,
                 on_epoch=True)
        self.log("Test Perimeter Difference Ophtal 2: ", np.nan_to_num(dif_peri_out_ophtal2.mean().detach().cpu().numpy()),
                 on_step=False, on_epoch=True)
        self.log("Test PCC Area Ophtal 2: ", torch.nan_to_num(self.PCC(area_calculation(y_ophtal_2).cuda() * 0.00017936,
                                                              area_calculation(
                                                                  image_out_clean_list).cuda() * 0.00017936)),
                 on_step=False, on_epoch=True)
        self.log("Test PCC Perimiter Ophtal 2: ", torch.nan_to_num(self.PCC(perimiter_calculation(y_ophtal_2.cuda()).cuda() * 0.01339,
                                                                   perimiter_calculation(
                                                                       image_out_clean_list.cuda()).cuda() * 0.01339)),
                 on_step=False, on_epoch=True)
        self.log("Test PCC Roundness Ophtal 2: ", torch.nan_to_num(
            self.PCC(roundness_calculation(y_ophtal_2).cuda(), roundness_calculation(image_out_clean_list).cuda())),
                 on_step=False, on_epoch=True)

        return test_loss

    def forward(self, x, num):
        bground = self.model(x.float())
        image_out = torch.where(bground.float() >= 0.5, 1, 0)
        image_out = image_out > 0.5
        image_out_clean_list = torch.ones((1, 1, image_out.shape[2], image_out.shape[3]))
        transform = T.ToPILImage()
        transform(torch.tensor(image_out_clean_list * 255, dtype=torch.uint8).squeeze(0)).save(
            self.image_path + "test_batch_bg_faz_seg_" + str(num) + "_sonia_" + str(self.learning_rate) + "_" + str(
                self.batch_size) + "_batch_ind_" + str(num) + ".tiff", quality=100)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.learning_rate)
        return optimizer

class PreTrain_ImReconstruct(pl.LightningModule):
    def __init__(self, model, batch_size, learning_rate, model_path):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.lr = learning_rate
        self.model_path = model_path
        self.PNSR = PeakSignalNoiseRatio(1)
        self.TV = TotalVariation()
        self.SSIM = StructuralSimilarityIndexMeasure()
        self.FID = FrechetInceptionDistance(feature=64)
        self.L1 = nn.L1Loss()
        self.loss = nn.MSELoss()

        self.best_mse_loss = 10**12
        self.best_SSIM_loss = 0
        self.best_FID_loss = 10**12
        self.best_PSNR_loss = 0
        self.best_debug_init_loss = 10**12
    
    def training_step(self, batch, batch_idx):
        image_defect, image, final_mask, self.dataset_index = batch
        bg_image = self.model(image_defect)
        bg_image = F.sigmoid(bg_image)
        self.FID.update(torch.tensor(torch.tile(bg_image, (1, 3, 1, 1))*255, dtype=torch.uint8), real=True)
        self.FID.update(torch.tensor((image*final_mask*255), dtype=torch.uint8), real=False)
        FID_value = torch.nan_to_num(self.FID.compute())

        if self.best_FID_loss > FID_value:
            self.log("best_FID: ",
                 FID_value, on_step=False,
                 on_epoch=True)
            save_image_batch_3chan_pre_train(bg_image[:5].detach().cpu(), image_defect[:5].detach().cpu(), image[:5], "segmentation_models/image_impaint/pre_train_FID_ims_", self.lr, self.batch_size)
        if self.best_SSIM_loss > torch.nan_to_num(self.SSIM(torch.tile(bg_image, (1, 3, 1, 1))*255, image*final_mask)):
            self.log("best_SSIM: ",
                 torch.nan_to_num(self.SSIM(torch.tile(bg_image, (1, 3, 1, 1)), image*final_mask)), on_step=False,
                 on_epoch=True)
        if self.best_PSNR_loss > torch.nan_to_num(self.PNSR(torch.tile(bg_image, (1, 3, 1, 1))*255, image*final_mask)):
            self.log("best_PNSR: ",
                 torch.nan_to_num(self.PNSR(torch.tile(bg_image, (1, 3, 1, 1)), image*final_mask)), on_step=False,
                 on_epoch=True)
        if self.best_debug_init_loss > torch.nan_to_num(self.PNSR(torch.tile(bg_image, (1, 3, 1, 1)), image*final_mask)):
            self.log("best_MSE_mask_impaint: ",
                 torch.nan_to_num(self.loss(torch.tile(bg_image, (1, 3, 1, 1)), image*final_mask)), on_step=False,
                 on_epoch=True)

        self.log("FID: ",
                 FID_value, on_step=False,
                 on_epoch=True)
        self.log("L1 Loss: ",
                 torch.nan_to_num(self.L1(torch.tile(bg_image, (1, 3, 1, 1)), image*final_mask)), on_step=False,
                 on_epoch=True)
        self.log("MSE overall Loss: ",
                 torch.nan_to_num(self.loss(torch.tile(bg_image, (1, 3, 1, 1)), image)), on_step=False,
                 on_epoch=True)
        self.log("MSE_mask_impaint: ", torch.nan_to_num(self.loss(torch.tile(bg_image, (1, 3, 1, 1)), image*final_mask)),
                 on_step=False, on_epoch=True)
        self.log("Peak Signal Noise Ratio: ",
                 torch.nan_to_num(self.PNSR(torch.tile(bg_image, (1, 3, 1, 1)), image*final_mask)), on_step=False,
                 on_epoch=True)
        self.log("SSIM: ",
                 torch.nan_to_num(self.SSIM(torch.tile(bg_image, (1, 3, 1, 1)), image*final_mask)), on_step=False,
                 on_epoch=True)
        self.log("Total Variation: ",
                 torch.nan_to_num(self.TV(torch.tile(bg_image, (1, 3, 1, 1)))), on_step=False,
                 on_epoch=True)
        self.log("debug_init_loss: ",
                 torch.nan_to_num(self.loss(torch.tile(bg_image, (1, 3, 1, 1)), image*final_mask)), on_step=False,
                 on_epoch=True)
        return self.loss(bg_image*final_mask, image*final_mask)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
                                params=self.model.parameters(), 
                                lr=self.lr)
        return optimizer
