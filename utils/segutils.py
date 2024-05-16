import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import torch.nn.functional as F

norm = lambda t: (t - t.min()) / (t.max() - t.min())
denorm = lambda t, min_, max_: t * (max_ - min_) + min_

percentilerange = lambda t, perc: t.min() + perc * (t.max() - t.min())
midrange = lambda t: percentilerange(t, .5)

downsample_mask = lambda mask, H, W: F.interpolate(mask.unsqueeze(1), size=(H, W), mode='bilinear',
                                                   align_corners=False).squeeze(1)


# downsampled_mask: [bsz,vecs], vecs can be H*W for example
# s_feat_volume: [bsz,c,vecs]
# returns [bsz,c], [bsz,c,vecs]
def fg_bg_proto(sfeat_volume, downsampled_smask):
    B, C, vecs = sfeat_volume.shape
    reshaped_mask = downsampled_smask.expand(B, vecs).unsqueeze(1)  # ->[B,1,vecs]

    masked_fg = reshaped_mask * sfeat_volume
    fg_proto = torch.sum(masked_fg, dim=-1) / (torch.sum(reshaped_mask, dim=-1) + 1e-8)

    masked_bg = (1 - reshaped_mask) * sfeat_volume
    bg_proto = torch.sum(masked_bg, dim=-1) / (torch.sum(1 - reshaped_mask, dim=-1) + 1e-8)
    assert fg_proto.shape == (B, C), ":o"
    return fg_proto, bg_proto


intersection = lambda pred, target: (pred * target).float().sum()
union = lambda pred, target: (pred + target).clamp(0, 1).float().sum()

def iou(pred, target):  # binary only, input bsz,h,w
    i, u = intersection(pred, target), union(pred, target)
    iou = (i + 1e-8) / (u + 1e-8)
    return iou.item()

#
# class SimpleAvgMeter:
#     def __init__(self, n_classes, device=torch.device('cuda')):
#         self.n_lasses = n_classes
#         self.intersection_buf = torch.zeros(n_classes).to(device)
#         self.union_buf = torch.zeros(n_classes).to(device)
#
#     def update(self, pred, target, class_id):
#         self.intersection_buf[class_id] += intersection(pred, target)
#         self.union_buf[class_id] += union(pred, target)
#
#     def IoU(self, class_id):
#         return self.intersection_buf[class_id] / self.union_buf[class_id] * 100
#
#     def cls_mIoU(self, class_ids):
#         return (self.intersection_buf[class_ids] / self.union_buf[class_ids]).mean() * 100
#
#     def compute_mIoU(self):
#         noentry = self.union_buf == 0
#         if noentry.sum() > 0: print("SimpleAvgMeter warning: ", noentry.sum(), "elements of", self.nclasses,
#                                     "have no empty.")
#         return self.cls_mIoU(~noentry)

# class KMeans():
#     # expects input to be in shape [bsz, -1]
#     def __init__(self, data, k=2, num_iterations=10):
#         self.k = k
#         self.device = data.device
#         self.centroids = self._init_centroids(data)
#
#         for _ in range(num_iterations):
#             labels = self._assign_clusters(data)
#             self._update_centroids(data, labels)
#
#         self.labels = self._assign_clusters(data)  # Final cluster assignment
#
#     def _init_centroids(self, data):
#         # Randomly initialize centroids
#         centroids = []
#         min_values = data.min(dim=1, keepdim=True).values
#         range_values = (data.max(dim=1, keepdim=True).values - min_values)
#
#         for _ in range(self.k):
#             random_values = torch.rand((data.shape[0], 1)).to(self.device)
#             centroids.append(min_values + random_values * range_values)
#
#         return torch.cat(centroids, dim=1)
#
#     def _assign_clusters(self, data):
#         # Calculate distances between data points and centroids
#         distances = torch.abs(data.unsqueeze(2) - self.centroids)  # Expand data tensor to calculate distances
#         # Determine the closest centroid for each data point
#         labels = torch.argmin(distances, dim=2)
#         # Sort labels so that the largest mean data point has the highest label
#         cluster_means = [data[labels == k].mean() for k in range(self.k)]
#         sorted_labels = {k: rank for rank, k in enumerate(sorted(range(self.k), key=lambda k: cluster_means[k]))}
#         labels = torch.tensor([sorted_labels[label.item()] for label in labels.flatten()]).reshape_as(labels).to(
#             self.device)
#
#         return labels
#
#     def _update_centroids(self, data, labels):
#         # Calculate new centroids as the mean of the data points closest to each centroid
#         mask = torch.nn.functional.one_hot(labels, num_classes=self.k).to(torch.float32)
#         summed_data = torch.bmm(mask.transpose(1, 2), data.unsqueeze(2))  # Sum data points per centroid
#         self.centroids = summed_data.squeeze() / mask.sum(dim=1, keepdim=True)  # Normalize to get the mean
#
#     def compute_thresholds(self):
#         # Flatten the centroids along the middle dimension
#         flat_centroids = self.centroids.view(self.centroids.size(0), -1)
#
#         # Sort the flattened centroids
#         sorted_centroids, _ = torch.sort(flat_centroids, dim=1)
#
#         # Compute the midpoints between consecutive centroids
#         thresholds = (sorted_centroids[:, :-1] + sorted_centroids[:, 1:]) / 2.0
#
#         return thresholds
#
#     def inference(self, data):
#         # Assign data points to the nearest centroid
#         return self._assign_clusters(data)

# def iterative_triclass_thresholding(image, max_iterations=100, tolerance=25):
#     # Ensure image is grayscale
#     if len(image.shape) == 3:
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # Initialize iteration parameters
#     TBD_region = image.copy()
#     iteration = 0
#     prev_threshold = 0
#
#     while iteration < max_iterations:
#         iteration += 1
#
#         # Step 1: Apply Otsu's thresholding on the TBD region
#         current_threshold, _ = cv2.threshold(TBD_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
#         # Check stopping criteria
#         if abs(current_threshold - prev_threshold) < tolerance:
#             break
#         prev_threshold = current_threshold
#
#         # Step 2: Calculate means for upper and lower regions
#         upper_region = TBD_region[TBD_region > current_threshold]
#         lower_region = TBD_region[TBD_region <= current_threshold]
#
#         if len(upper_region) == 0 or len(lower_region) == 0:
#             break  # No further division possible
#
#         mean_upper = np.mean(upper_region)
#         mean_lower = np.mean(lower_region)
#
#         # Step 3: Update temporary foreground, background, and TBD regions
#         TBD_region[(TBD_region > mean_upper)] = 255  # Temporary foreground F
#         TBD_region[(TBD_region < mean_lower)] = 0  # Temporary background B
#
#         # Extracting the new TBD region (between mean_lower and mean_upper)
#         mask = (TBD_region > mean_lower) & (TBD_region < mean_upper)
#         TBD_region = TBD_region[mask]  # Apply mask to extract region
#
#     # Final classification after convergence or max iterations
#     final_foreground = (image > current_threshold).astype(np.uint8) * 255
#     final_background = (image <= current_threshold).astype(np.uint8) * 255
#
#     return current_threshold, final_foreground

def otsus(batched_tensor_image, drop_least=0.05, mode='ordinary'):
    bsz = batched_tensor_image.size(0)
    binary_tensors = []
    thresholds = []

    for i in range(bsz):
        # Convert the tensor to numpy array
        numpy_image = batched_tensor_image[i].cpu().numpy()

        # Rescale to [0, 255] and convert to uint8 type for OpenCV compatibility
        npmin, npmax = numpy_image.min(), numpy_image.max()
        numpy_image = (norm(numpy_image) * 255).astype(np.uint8)

        # Drop values that are in the lowest percentiles
        truncated_vals = numpy_image[numpy_image >= int(255 * drop_least)]

        # Apply Otsu's thresholding
        if mode == 'via_triclass':
            thresh_value, _ = iterative_triclass_thresholding(truncated_vals)
        else:
            thresh_value, _ = cv2.threshold(truncated_vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Apply the computed threshold on the original image
        binary_image = (numpy_image > thresh_value).astype(np.uint8) * 255

        # Convert the result back to a tensor and append to the list
        binary_tensors.append(torch.from_numpy(binary_image).float() / 255)

        thresholds.append(torch.tensor(denorm(thresh_value / 255, npmin, npmax)) \
                          .to(batched_tensor_image.device, dtype=batched_tensor_image.dtype))

    # Convert list of tensors back to a single batched tensor
    binary_tensor_batch = torch.stack(binary_tensors, dim=0)
    thresh_batch = torch.stack(thresholds, dim=0)
    return thresh_batch, binary_tensor_batch


def iterative_otsus(probab_mask, s_mask, maxiters=5, mode='ordinary',
                    debug=False):  # verify that it works correctly when batch_size >1
    it = 1
    otsuthresh = 0
    assert probab_mask.min() >= 0 and probab_mask.max() <= 1, 'you should pass probabilites'
    while True:
        clipped = torch.where(probab_mask < otsuthresh, 0, probab_mask)
        otsuthresh, newmask = otsus(clipped.detach(), drop_least=.02, mode=mode)
        if otsuthresh >= s_mask.mean():
            return otsuthresh.to(probab_mask.device), newmask.to(probab_mask.device)
        if it >= maxiters:
            if debug:
                print('reached maxiter:', it, 'with thresh', otsuthresh.item(), \
                      'removed', int(((clipped == 0).sum() / clipped.numel()).item() * 10000) / 100, \
                      '% at lower and and new min,max is', clipped[clipped > 0].min().item(), clipped.max().item())
                display(pilImageRow(norm(probab_mask[0]), s_mask[0], maxwidth=300))
            return s_mask.mean(), (probab_mask > s_mask.mean()).float()  # otsuthresh
        it += 1


def calcthresh(fused_pred, s_masks, method='otsus'):
    #     if method=='iterotsus':
    #         thresh = iterative_otsus(fused_pred,s_masks,maxiters=5)[0]
    #         return thresh
    #     elif method=='1iterotsus':
    #         thresh = iterative_otsus(fused_pred,s_masks,maxiters=1)[0]
    #         return thresh
    if method == 'otsus':
        thresh = otsus(fused_pred)[0]
        return thresh
    elif method == 'pred_mean':
        otsu_thresh = otsus(fused_pred)[0]
        thresh = torch.max(otsu_thresh, fused_pred.mean())
    return thresh


def thresh_fn(method):
    def inner(fused_pred, s_masks=None):
        return calcthresh(fused_pred, s_masks, method)

    return inner

# def upgrade_scipy():
#     os.system('!pip install - -upgrade scipy')
#
#
# def slicRGB(q_img, n_segments=50, compactness=10., sigma=1, mask=None, debug=False):
#     import skimage.segmentation as skseg
#
#     rgb_labels = skseg.slic(q_img, n_segments=n_segments, compactness=compactness, sigma=sigma, mask=mask,
#                             enforce_connectivity=True)
#
#     if debug:
#         plt.imshow(skseg.mark_boundaries(q_img, rgb_labels))
#         plt.show()
#
#     return rgb_labels
#
#
#
# def slicRGBP(q_img, fg_pred, n_segments=30, compactness=0.1, sigma=1, mask=None, debug=False):
#     import skimage.segmentation as skseg
#
#     def concat_rgb_pred(rgbimg, pred):
#         h, w = rgbimg.shape[:2]
#         return np.concatenate((rgbimg, pred.reshape(h, w, 1)), axis=-1)
#
#     rgbp_img = concat_rgb_pred(q_img, fg_pred)
#     rgbp_labels = skseg.slic(rgbp_img, n_segments=n_segments, compactness=compactness, mask=mask, sigma=sigma,
#                              enforce_connectivity=True)
#
#     if debug:
#         rgb_labels = skseg.slic(q_img, n_segments=n_segments, compactness=10., sigma=sigma, mask=mask,
#                                 enforce_connectivity=True)
#         pred_labels = skseg.slic(fg_pred, n_segments=n_segments, compactness=compactness, sigma=sigma, mask=mask,
#                                  channel_axis=None, enforce_connectivity=True)
#
#         rows, cols = 1, 3
#         fig, ax = plt.subplots(rows, cols, figsize=(10, 10), sharex=True, sharey=True)
#         ax[0].imshow(skseg.mark_boundaries(q_img, rgbp_labels))
#         ax[1].imshow(skseg.mark_boundaries(q_img, rgb_labels))
#         ax[2].imshow(skseg.mark_boundaries(q_img, pred_labels))
#         plt.show()
#
#     return rgbp_labels
#
#
# def calc_cluster_means(label_id_map, fg_prob):
#     fg_pred_clustered = np.zeros_like(fg_prob)
#     label_ids = np.unique(label_id_map)
#     for lab_id in label_ids:
#         cluster = fg_prob[label_id_map == lab_id]
#         fg_pred_clustered[label_id_map == lab_id] = cluster.mean()
#     return fg_pred_clustered


def install_pydensecrf():
    os.system('pip install git+https://github.com/lucasb-eyer/pydensecrf.git')


class CRF:
    def __init__(self, gaussian_stdxy=(3, 3), gaussian_compat=3,
                 bilateral_stdxy=(80, 80), bilateral_compat=10, stdrgb=(13, 13, 13)):
        self.gaussian_stdxy = gaussian_stdxy
        self.gaussian_compat = gaussian_compat
        self.bilateral_stdxy = bilateral_stdxy
        self.bilateral_compat = bilateral_compat
        self.stdrgb = stdrgb
        self.iters = 5
        self.debug = False

    def refine(self, image_tensor, fg_probs, soft_thresh=None, T=1):

        """
        Refine segmentation using DenseCRF.

        Args:
            - image_tensor (tensor): Original image, shape [1, 3, H, W].
            - fg_probs (tensor): Fg probabilities from the network, shape [1, H, W]
            - soft_thresh: The preferred threshold for fg_probs for segmenting into binary prediction mask
            - T: a temperature for softmax/sigmoid

        Returns:
            - Refined segmentation mask, shape [1, H, W].
        """
        try:
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
        except ImportError as e:
            print("pydensecrf not found. Installing...")
            install_pydensecrf()  # Ensure this function installs pydensecrf and handles any potential errors during installation.

        # After installation, retry importing. This is placed inside the except block to avoid repeating the import statements.
        try:
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
        except ImportError as e:
            print("Failed to import after installation. Please check the installation of pydensecrf.")
            raise  # This will raise the last exception that was handled by the except block

        # We find the segmentation threshold that splits fg-bg
        if soft_thresh is None:
            soft_thresh, _ = otsus(fg_probs)
        image_tensor, fg_probs, soft_thresh = image_tensor.cpu(), fg_probs.cpu(), soft_thresh.cpu()
        # Then we presume at this threshold the probability should be 0.5
        # probability 0 should stay 0, 1 should stay 1
        # sigmoid=lambda x: 1/(1 + np.exp(-x))
        fg_probs = torch.sigmoid(T * (fg_probs - soft_thresh))
        probs = torch.stack([1 - fg_probs, fg_probs], dim=1)  # crf expects both classes as input
        if self.debug:
            print('softthresh', soft_thresh)
            print('fg_probs min max', fg_probs.min(), fg_probs.max())
        # C: Number of classes
        bsz, C, H, W = probs.shape

        refined_masks = []
        image_numpy = np.ascontiguousarray( \
            (255 * image_tensor.permute(0, 2, 3, 1)).numpy().astype(np.uint8))
        probs_numpy = probs.numpy()
        for (image, prob) in zip(image_numpy, probs_numpy):
            # Unary potentials
            unary = np.ascontiguousarray(unary_from_softmax(prob))
            d = dcrf.DenseCRF2D(W, H, C)
            d.setUnaryEnergy(unary)

            # Add pairwise potentials
            d.addPairwiseGaussian(sxy=self.gaussian_stdxy, compat=self.gaussian_compat)
            d.addPairwiseBilateral(sxy=self.bilateral_stdxy, srgb=self.stdrgb,
                                   rgbim=image, compat=self.bilateral_compat)

            # Perform inference
            Q = d.inference(self.iters)
            if self.debug:
                print('Q:', np.array(Q).shape, np.array(Q)[0].mean(), np.array(Q).mean())
            result = np.reshape(Q, (2, H, W))  # np.argmax(Q, axis=0).reshape((H, W))
            refined_masks.append(result)

        return torch.from_numpy(np.stack(refined_masks, axis=0))

    def iterrefine(self, iters, q_img, fg_probs, thresh_fn, debug=False):
        pred = fg_probs.unsqueeze(1).expand(1, 2, *fg_probs.shape[-2:])
        for it in range(iters):
            thresh = thresh_fn(pred[:, 1])[0]

            if debug and i % 10 == 0:
                print('thresh', thresh)
                display(to_pil(pred[0, 1]))

            pred = self.refine(q_img, pred[:, 1], soft_thresh=thresh)
        return pred

#
# class Subplot:
#     def __init__(self):
#         self.vertical_lines = []
#         self.histograms = []
#         self.gaussian_curves = []
#         self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
#         self.title = ''
#
#     class Element:
#         def __init__(self, x=None, y=None, label=''):
#             if x is not None:
#                 self.x = Subplot.to_np(x)
#             if y is not None:
#                 self.y = Subplot.to_np(y)
#
#             self.label = label
#
#     @staticmethod
#     def to_np(t):
#         return t.detach().cpu().numpy()
#
#     def add_vertical(self, x, label=''):
#         self.vertical_lines.append(Subplot.Element(x=x, label=label))
#         return self
#
#     def add_histogram(self, samples, label=''):
#         self.histograms.append(Subplot.Element(x=samples, label=label))
#         return self
#
#     def add_gaussian(self, gaussian):
#         samples, mu, var = gaussian.samples, gaussian.mean, gaussian.covs
#         # Generate a range of x values
#         x_values = np.linspace(samples.min(), samples.max(), 100)
#         x_values = np.linspace(samples.min(), samples.max(), 100)
#
#         # Compute Gaussian values for these x values
#         gaussian1_values = gaussian.gaussian_pdf(x_values, mu[0].item(), var[0].item())
#         gaussian2_values = gaussian.gaussian_pdf(x_values, mu[1].item(), var[1].item())
#         self.gaussian_curves.append(Subplot.Element(x_values, gaussian1_values))
#         self.gaussian_curves.append(Subplot.Element(x_values, gaussian2_values))
#         return self
#
#
# class PredHistos2():
#     def __init__(self, n_cols=1):
#         self.fig, self.axes = plt.subplots(nrows=1, ncols=n_cols, figsize=(10, 4))
#         self.n_cols = n_cols
#         if n_cols == 1:
#             self.builder = Subplot()
#         self.subplots = [Subplot() for x in range(n_cols)]
#         self.alpha = 0.5
#         self.bins = 200
#
#     def reload(self, n_cols=1):
#         self.fig, self.axes = plt.subplots(nrows=1, ncols=n_cols, figsize=(10, 4))
#
#     def aggr(self, ax, sub):
#         for hist, col in zip(sub.histograms, sub.colors):
#             ax.hist(hist.x, self.bins, density=True, color=col, alpha=self.alpha, label=hist.label)
#         for vline, col in zip(sub.vertical_lines, sub.colors):
#             ax.axvline(x=vline.x, color=col, label=vline.label, linestyle='--')
#         for gaussian, col in zip(sub.gaussian_curves, sub.colors):
#             ax.plot(gaussian.x, gaussian.y, gaussian.label, col)
#         ax.legend()
#
#     def plot(self, name=''):
#
#         if self.n_cols == 1:
#             self.aggr(plt, self.builder)
#         else:
#             for ax, sub in zip(self.axes, self.subplots):
#                 self.aggr(ax, sub)
#                 ax.set_title(sub.title)
#
#         plt.legend()
#         plt.title(name)
#         plt.show()
#
#
# from sklearn.mixture import GaussianMixture
# import scipy.optimize as opt
# from scipy.optimize import fsolve
#
#
# class GMM:
#     def __init__(self, q_pred_coarse, name='gaussian', n_components=2):
#         samples = q_pred_coarse.detach().cpu().numpy()
#         self.samples = samples.reshape(-1, 1)
#
#         # Fit a mixture of 2 Gaussians using EM
#         gmm = GaussianMixture(n_components)
#         gmm.fit(samples)
#         self.means = gmm.means_.flatten()
#         self.covs = gmm.covariances_.flatten()
#         self.weights = gmm.weights_
#         self.label = name
#
#     def intersect(self):
#         # Use fsolve to find the intersection
#         gaussian_intersect, = fsolve(difference, self.means.mean(), args=(
#         self.means[0].item(), self.covs[0].item(), self.means[1].item(), self.means[1].item()))
#         return gaussian_intersect
#
#
# class PredHistoSNS():
#     def __init__(self, n_cols=1):
#         import seaborn as sns
#         sns.set_theme(style="whitegrid")  # Set the Seaborn theme. You can change the style as needed.
#         self.fig, self.axes = plt.subplots(nrows=1, ncols=n_cols, figsize=(10, 4))
#         self.n_cols = n_cols
#         if n_cols == 1:
#             self.axes = [self.axes]  # Wrap the single axis in a list to simplify the loop logic later.
#             self.builder = Subplot()  # This is assuming Subplot is a properly defined class.
#         self.subplots = [Subplot() for _ in range(n_cols)]  # Use underscore for unused loop variable.
#         self.alpha = 0.5
#         self.bins = 200
#
#     def reload(self, n_cols=1):
#         self.fig, self.axes = plt.subplots(nrows=1, ncols=n_cols, figsize=(10, 4))
#
#     def aggr(self, ax, sub):
#         import seaborn as sns
#         for hist, col in zip(sub.histograms, sub.colors):
#             sns.histplot(hist.x, bins=self.bins, kde=False, color=col, ax=ax, alpha=self.alpha, label=hist.label)
#         for vline, col in zip(sub.vertical_lines, sub.colors):
#             ax.axvline(x=vline.x, color=col, label=vline.label, linestyle='--')
#         for gaussian, col in zip(sub.gaussian_curves, sub.colors):
#             sns.lineplot(x=gaussian.x, y=gaussian.y, label=gaussian.label, color=col, ax=ax)
#         ax.legend()
#
#     def plot(self, name=''):
#
#         if self.n_cols == 1:
#             self.aggr(self.axes[0], self.builder)
#         else:
#             for ax, sub in zip(self.axes, self.subplots):
#                 self.aggr(ax, sub)
#                 ax.set_title(sub.title)
#
#         plt.show()
#
#
# def overlay_mask(image, mask, color=[255, 0, 0], alpha=0.2):
#     """
#     Apply an overlay of a binary mask onto an image using a specified color.
#
#     :param image: A PyTorch tensor of the image (C x H x W) with pixel values in [0, 1].
#     :param mask: A PyTorch tensor of the mask (H x W) with binary values (0 or 1).
#     :param color: A list of 3 elements representing the RGB values of the overlay color.
#     :param alpha: A float representing the transparency of the overlay (0 to 1).
#     :return: An overlayed image tensor.
#     """
#     # Ensure the mask is binary
#     mask = (mask > 0).float()
#
#     # Create an RGB version of the mask
#     mask_rgb = torch.tensor(color).view(3, 1, 1) / 255.0  # Normalize the color vector
#     mask_rgb = mask_rgb * mask
#
#     # Overlay the mask onto the image
#     overlayed_image = (1 - alpha) * image + alpha * mask_rgb
#
#     # Ensure the resulting tensor values are between 0 and 1
#     overlayed_image = torch.clamp(overlayed_image, 0, 1)
#
#     return overlayed_image
#

import pandas as pd

to_pil = lambda t: transforms.ToPILImage()(t) if t.shape[-1] > 4 else transforms.ToPILImage()(t.permute(2, 0, 1))


def pilImageRow(*imgs, maxwidth=800, bordercolor=0x000000):
    imgs = [to_pil(im.float()) for im in imgs]
    dst = Image.new('RGB', (sum(im.width for im in imgs), imgs[0].height))
    for i, im in enumerate(imgs):
        loc = [x0, y0, x1, y1] = [i * im.width, 0, (i + 1) * im.width, im.height]
        dst.paste(im, (x0, y0))
        ImageDraw.Draw(dst).rectangle(loc, width=2, outline=bordercolor)
    factorToBig = dst.width / maxwidth
    dst = dst.resize((int(dst.width / factorToBig), int(dst.height / factorToBig)))
    return dst


def tensor_table(**kwargs):
    tensor_overview = {}
    for name, tensor in kwargs.items():
        if callable(tensor):
            print(name, [tensor(t) for _, t in kwargs.items() if isinstance(t, torch.Tensor)])
        else:
            tensor_overview[name] = {
                'min': tensor.min().item(),
                'max': tensor.max().item(),
                'shape': tensor.shape,
            }
    return pd.DataFrame.from_dict(tensor_overview, orient='index')

