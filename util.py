from PIL import Image
import numpy as np
import random
import math
import os
import skimage.measure
import skimage.color


def recursive_forwarding(LR, scale, chopSize, session, net_model, chopShave = 10):
    b, h, w, c = LR.shape
    wHalf = math.floor(w / 2)
    hHalf = math.floor(h / 2)
    
    wc = wHalf + chopShave
    hc = hHalf + chopShave
    
    inputPatch = np.array((LR[:, :hc, :wc, :], LR[:, :hc, (w-wc):, :], LR[:,(h-hc):,:wc,:], LR[:,(h-hc):,(w-wc):,:]))
    outputPatch = []

    if wc * hc < chopSize:
        for ele in inputPatch:
            output = session.run(net_model.output, feed_dict = {net_model.LR : ele})
            outputPatch.append(output)

    else:
        for ele in inputPatch:
            output = recursive_forwarding(ele, scale, chopSize, session, net_model, chopShave)
            outputPatch.append(output)

    w = scale * w
    wHalf = scale * wHalf
    wc = scale * wc
    
    h = scale * h
    hHalf = scale * hHalf
    hc = scale * hc
    
    chopShave = scale * chopSize
    
    upper = np.concatenate((outputPatch[0][:,:hHalf,:wHalf,:], outputPatch[1][:,:hHalf,wc-w+wHalf:,:]), axis = 2)
    rower = np.concatenate((outputPatch[2][:,hc-h+hHalf:,:wHalf,:], outputPatch[3][:,hc-h+hHalf:,wc-w+wHalf:,:]), axis = 2)
    output = np.concatenate((upper,rower),axis = 1)
    
    return output
    
def self_ensemble(args, model, sess, LR_img, is_recursive = False):
    
    input_img_list = []
    output_img_list = []
    for i in range(4):
        input_img_list.append(np.rot90(LR_img, i))
    input_img_list.append(np.fliplr(LR_img))
    input_img_list.append(np.flipud(LR_img))
    input_img_list.append(np.rot90(np.fliplr(LR_img)))
    input_img_list.append(np.rot90(np.flipud(LR_img)))
    
    
    for ele in input_img_list:
        
        input_img = np.expand_dims(ele, axis = 0)
        
        if is_recursive :
            output_img = recursive_forwarding(input_img, args.scale, args.chop_size, sess, model, args.chop_shave)
            output_img_list.append(output_img[0])
            
        else:
            output_img = sess.run(model.output, feed_dict = {model.LR : input_img})
            output_img_list.append(output_img[0])
            
    reshaped_output_img_list = []
    for i in range(4):
        output_img = output_img_list[i]
        output_img = np.rot90(output_img, 4-i)
        output_img = output_img.astype(np.float32)
        reshaped_output_img_list.append(output_img)
    
    output_img = output_img_list[4]
    output_img = np.fliplr(output_img)
    output_img = output_img.astype(np.float32)
    reshaped_output_img_list.append(output_img)
    
    output_img = output_img_list[5]
    output_img = np.flipud(output_img)
    output_img = output_img.astype(np.float32)
    reshaped_output_img_list.append(output_img)
    
    output_img = output_img_list[6]
    output_img = np.fliplr(np.rot90(output_img,3))
    output_img = output_img.astype(np.float32)
    reshaped_output_img_list.append(output_img)
    
    output_img = output_img_list[7]
    output_img = np.flipud(np.rot90(output_img,3))
    output_img = output_img.astype(np.float32)
    reshaped_output_img_list.append(output_img)
    
    overall_img = sum(reshaped_output_img_list) / 8
    overall_img = np.clip(np.round(overall_img), 0.0, 255.0)
    overall_img = overall_img.astype(np.uint8)
        
    return overall_img

def image_loader(image_path):
    
    imgs = sorted(os.listdir(image_path))
    img_list = []
    for ele in imgs:
        img_list.append(np.array(Image.open(os.path.join(image_path, ele))))
    
    return np.array(img_list)

def data_augument(lr_img, hr_img, aug):
    
    if aug < 4:
        lr_img = np.rot90(lr_img, aug)
        hr_img = np.rot90(hr_img, aug)
    
    elif aug == 4:
        lr_img = np.fliplr(lr_img)
        hr_img = np.fliplr(hr_img)
        
    elif aug == 5:
        lr_img = np.flipud(lr_img)
        hr_img = np.flipud(hr_img)
        
    elif aug == 6:
        lr_img = np.rot90(np.fliplr(lr_img))
        hr_img = np.rot90(np.fliplr(hr_img))
        
    elif aug == 7:
        lr_img = np.rot90(np.flipud(lr_img))
        hr_img = np.rot90(np.flipud(hr_img))
        
    return lr_img, hr_img

def batch_gen(LR_imgs, HR_imgs, patch_size, scale, batch_size, random_index, step):
    
    all_img_lr = LR_imgs[random_index[step * batch_size : (step + 1) * batch_size]]
    all_img_hr = HR_imgs[random_index[step * batch_size : (step + 1) * batch_size]]
    
    lr_batch = []
    hr_batch = []
    
    for i in range(len(all_img_lr)):
        
        ih, iw, _ = all_img_lr[i].shape
        ix = random.randrange(0, iw - patch_size +1)
        iy = random.randrange(0, ih - patch_size +1)
        
        tx = ix * scale
        ty = iy * scale
        
        img_lr_in = all_img_lr[i][iy:iy + patch_size, ix:ix + patch_size]
        img_hr_in = all_img_hr[i][ty : ty + (scale * patch_size), tx : tx + (scale * patch_size)]        
        
        aug = random.randrange(0,8)
        
        img_lr_in, img_hr_in = data_augument(img_lr_in, img_hr_in, aug)
        
        lr_batch.append(img_lr_in)
        hr_batch.append(img_hr_in)
        
    lr_batch = np.array(lr_batch)
    hr_batch = np.array(hr_batch)
    
    return lr_batch, hr_batch


def train_with_test(args, model, sess, saver, f, count, val_lr_imgs = None, val_gt_imgs = None, val_num = 0):
    
    RGB_PSNR_list = []
    Y_PSNR_list = []
    
    if args.in_memory:
    
        for i, lr_img in enumerate(val_lr_imgs):

            resize_lr_img = np.expand_dims(lr_img, axis = 0)
            output, learning_rate = sess.run([model.output, model.lr], feed_dict = {model.LR : resize_lr_img, model.global_step : count})
            output = output[0]

            h, w, c = output.shape
            val_gt = val_gt_imgs[i][:h,:w,:]

            rgb_psnr = skimage.measure.compare_psnr(output[args.scale:-args.scale, args.scale:-args.scale,:], val_gt[args.scale:-args.scale, args.scale:-args.scale,:], data_range = 255)
            y_output = skimage.color.rgb2ycbcr(output)
            y_gt = skimage.color.rgb2ycbcr(val_gt)
            y_output = y_output / 255.0
            y_gt = y_gt / 255.0
            y_psnr = skimage.measure.compare_psnr(y_output[args.scale:-args.scale, args.scale:-args.scale, :1], y_gt[args.scale:-args.scale, args.scale:-args.scale, :1], data_range = 1.0)

            RGB_PSNR_list.append(rgb_psnr)
            Y_PSNR_list.append(y_psnr)
    else:
        
        sess.run(model.data_loader.init_op['val_init'])
        num = len(os.listdir(args.train_GT_path))
        
        for i in range(num):
            output, val_gt, learning_rate = sess.run(model.output, model.label, model.lr, feed_dict = {model.global_step : count})
            output = output[0]
            val_gt = val_gt[0]
            h, w, c = output.shape
            val_gt = val_gt[:h,:w]

            rgb_psnr = skimage.measure.compare_psnr(output[args.scale:-args.scale, args.scale:-args.scale,:], val_gt[args.scale:-args.scale, args.scale:-args.scale,:], data_range = 255)
            y_output = skimage.color.rgb2ycbcr(output)
            y_gt = skimage.color.rgb2ycbcr(val_gt)
            y_output = y_output / 255.0
            y_gt = y_gt / 255.0
            
            y_psnr = skimage.measure.compare_psnr(y_output[args.scale:-args.scale, args.scale:-args.scale, :1], y_gt[args.scale:-args.scale, args.scale:-args.scale, :1], data_range = 1.0)

            RGB_PSNR_list.append(rgb_psnr)
            Y_PSNR_list.append(y_psnr)            

        
    mean_RGB_PSNR = np.mean(RGB_PSNR_list)
    mean_Y_PSNR = np.mean(Y_PSNR_list)

    f.write('%06d-count \t lr : %04f \t RGB_PSNR : %04f \t Y_PSNR : %04f \n'%(count, learning_rate, mean_RGB_PSNR, mean_Y_PSNR))

import numpy as np
from numpy.lib.arraypad import _validate_lengths
from scipy.ndimage import uniform_filter, gaussian_filter

dtype_range = {np.bool_: (False, True),
               np.bool8: (False, True),
               np.uint8: (0, 255),
               np.uint16: (0, 65535),
               np.uint32: (0, 2**32 - 1),
               np.uint64: (0, 2**64 - 1),
               np.int8: (-128, 127),
               np.int16: (-32768, 32767),
               np.int32: (-2**31, 2**31 - 1),
               np.int64: (-2**63, 2**63 - 1),
               np.float16: (-1, 1),
               np.float32: (-1, 1),
               np.float64: (-1, 1)}


def crop(ar, crop_width, copy=False, order='K'):
    """Crop array `ar` by `crop_width` along each dimension.
    Parameters
    ----------
    ar : array-like of rank N
        Input array.
    crop_width : {sequence, int}
        Number of values to remove from the edges of each axis.
        ``((before_1, after_1),`` ... ``(before_N, after_N))`` specifies
        unique crop widths at the start and end of each axis.
        ``((before, after),)`` specifies a fixed start and end crop
        for every axis.
        ``(n,)`` or ``n`` for integer ``n`` is a shortcut for
        before = after = ``n`` for all axes.
    copy : bool, optional
        If `True`, ensure the returned array is a contiguous copy. Normally,
        a crop operation will return a discontiguous view of the underlying
        input array.
    order : {'C', 'F', 'A', 'K'}, optional
        If ``copy==True``, control the memory layout of the copy. See
        ``np.copy``.
    Returns
    -------
    cropped : array
        The cropped array. If ``copy=False`` (default), this is a sliced
        view of the input array.
    """
    ar = np.array(ar, copy=False)
    crops = _validate_lengths(ar, crop_width)
    slices = [slice(a, ar.shape[i] - b) for i, (a, b) in enumerate(crops)]
    if copy:
        cropped = np.array(ar[slices], order=order, copy=True)
    else:
        cropped = ar[slices]
    return cropped

def compare_ssim(X, Y, win_size=None, gradient=False,
                 data_range=None, multichannel=False, gaussian_weights=False,
                 full=False, dynamic_range=None, **kwargs):
    """Compute the mean structural similarity index between two images.
    Parameters
    ----------
    X, Y : ndarray
        Image.  Any dimensionality.
    win_size : int or None
        The side-length of the sliding window used in comparison.  Must be an
        odd value.  If `gaussian_weights` is True, this is ignored and the
        window size will depend on `sigma`.
    gradient : bool, optional
        If True, also return the gradient.
    data_range : int, optional
        The data range of the input image (distance between minimum and
        maximum possible values).  By default, this is estimated from the image
        data-type.
    multichannel : bool, optional
        If True, treat the last dimension of the array as channels. Similarity
        calculations are done independently for each channel then averaged.
    gaussian_weights : bool, optional
        If True, each patch has its mean and variance spatially weighted by a
        normalized Gaussian kernel of width sigma=1.5.
    full : bool, optional
        If True, return the full structural similarity image instead of the
        mean value.
    Other Parameters
    ----------------
    use_sample_covariance : bool
        if True, normalize covariances by N-1 rather than, N where N is the
        number of pixels within the sliding window.
    K1 : float
        algorithm parameter, K1 (small constant, see [1]_)
    K2 : float
        algorithm parameter, K2 (small constant, see [1]_)
    sigma : float
        sigma for the Gaussian when `gaussian_weights` is True.
    Returns
    -------
    mssim : float
        The mean structural similarity over the image.
    grad : ndarray
        The gradient of the structural similarity index between X and Y [2]_.
        This is only returned if `gradient` is set to True.
    S : ndarray
        The full SSIM image.  This is only returned if `full` is set to True.
    Notes
    -----
    To match the implementation of Wang et. al. [1]_, set `gaussian_weights`
    to True, `sigma` to 1.5, and `use_sample_covariance` to False.
    References
    ----------
    .. [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
       (2004). Image quality assessment: From error visibility to
       structural similarity. IEEE Transactions on Image Processing,
       13, 600-612.
       https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
       DOI:10.1.1.11.2477
    .. [2] Avanaki, A. N. (2009). Exact global histogram specification
       optimized for structural similarity. Optical Review, 16, 613-621.
       http://arxiv.org/abs/0901.0065,
       DOI:10.1007/s10043-009-0119-z
    """
    if not X.dtype == Y.dtype:
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if dynamic_range is not None:
        #warn('`dynamic_range` has been deprecated in favor of '
        #     '`data_range`. The `dynamic_range` keyword argument '
        #     'will be removed in v0.14', skimage_deprecation)
        data_range = dynamic_range

    if multichannel:
        # loop over channels
        args = dict(win_size=win_size,
                    gradient=gradient,
                    data_range=data_range,
                    multichannel=False,
                    gaussian_weights=gaussian_weights,
                    full=full)
        args.update(kwargs)
        nch = X.shape[-1]
        mssim = np.empty(nch)
        if gradient:
            G = np.empty(X.shape)
        if full:
            S = np.empty(X.shape)
        for ch in range(nch):
            ch_result = compare_ssim(X[..., ch], Y[..., ch], **args)
            if gradient and full:
                mssim[..., ch], G[..., ch], S[..., ch] = ch_result
            elif gradient:
                mssim[..., ch], G[..., ch] = ch_result
            elif full:
                mssim[..., ch], S[..., ch] = ch_result
            else:
                mssim[..., ch] = ch_result
        mssim = mssim.mean()
        if gradient and full:
            return mssim, G, S
        elif gradient:
            return mssim, G
        elif full:
            return mssim, S
        else:
            return mssim

    K1 = kwargs.pop('K1', 0.01)
    K2 = kwargs.pop('K2', 0.03)
    sigma = kwargs.pop('sigma', 1.5)
    if K1 < 0:
        raise ValueError("K1 must be positive")
    if K2 < 0:
        raise ValueError("K2 must be positive")
    if sigma < 0:
        raise ValueError("sigma must be positive")
    use_sample_covariance = kwargs.pop('use_sample_covariance', True)

    if win_size is None:
        if gaussian_weights:
            win_size = 11  # 11 to match Wang et. al. 2004
        else:
            win_size = 7   # backwards compatibility

    if np.any((np.asarray(X.shape) - win_size) < 0):
        raise ValueError(
            "win_size exceeds image extent.  If the input is a multichannel "
            "(color) image, set multichannel=True.")

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    if data_range is None:
        dmin, dmax = dtype_range[X.dtype.type]
        data_range = dmax - dmin

    ndim = X.ndim

    if gaussian_weights:
        # sigma = 1.5 to approximately match filter in Wang et. al. 2004
        # this ends up giving a 13-tap rather than 11-tap Gaussian
        filter_func = gaussian_filter
        filter_args = {'sigma': sigma}

    else:
        filter_func = uniform_filter
        filter_args = {'size': win_size}

    # ndimage filters need floating point data
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)

    NP = win_size ** ndim

    # filter has already normalized by NP
    if use_sample_covariance:
        cov_norm = NP / (NP - 1)  # sample covariance
    else:
        cov_norm = 1.0  # population covariance to match Wang et. al. 2004

    # compute (weighted) means
    ux = filter_func(X, **filter_args)
    uy = filter_func(Y, **filter_args)

    # compute (weighted) variances and covariances
    uxx = filter_func(X * X, **filter_args)
    uyy = filter_func(Y * Y, **filter_args)
    uxy = filter_func(X * Y, **filter_args)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    R = data_range
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2

    A1, A2, B1, B2 = ((2 * ux * uy + C1,
                       2 * vxy + C2,
                       ux ** 2 + uy ** 2 + C1,
                       vx + vy + C2))
    D = B1 * B2
    S = (A1 * A2) / D

    # to avoid edge effects will ignore filter radius strip around edges
    pad = (win_size - 1) // 2

    # compute (weighted) mean of ssim
    mssim = crop(S, pad).mean()

    if gradient:
        # The following is Eqs. 7-8 of Avanaki 2009.
        grad = filter_func(A1 / D, **filter_args) * X
        grad += filter_func(-S / B2, **filter_args) * Y
        grad += filter_func((ux * (A2 - A1) - uy * (B2 - B1) * S) / D,
                            **filter_args)
        grad *= (2 / X.size)

        if full:
            return mssim, grad, S
        else:
            return mssim, grad
    else:
        if full:
            return mssim, S
        else:
            return mssim

