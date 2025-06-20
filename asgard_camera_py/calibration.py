import time
import numpy as np
from astropy.io import fits
from .shm_reader import SHMReader  # optional dependency
from .zmq_utils import CamClient, MDSClient

def build_manual_dark(camera, mds_client=None, no_frames=100, sleeptime=3, build_bad_pixel_mask=False, save_file_name=None):
    # Turn off source if MDS provided
    if mds_client:
        print("[MDS] Turning off source...")
        try:
            mds_client.send("off SBB")
            time.sleep(sleeptime)
        except Exception as e:
            print(f"[MDS] Warning: {e}")

    print("[CAL] Capturing dark frames...")
    dark_list = camera.get_some_frames(number_of_frames=no_frames, apply_manual_reduction=False)

    dark = np.mean(dark_list, axis=0).astype(int)

    if camera.pupil_crop_region == [None, None, None, None]:
        dark[0, 0:5] = np.mean(np.array(dark)[1:, 1:])

    if camera.reduction_dict['bias']:
        print("[CAL] Applying bias to dark...")
        dark -= camera.reduction_dict['bias'][-1]

    fps = float(camera.config["fps"])
    gain = float(camera.config["gain"])
    dark = (dark * fps / gain).astype(int)

    camera.reduction_dict['dark'].append(dark)

    if build_bad_pixel_mask:
        bad_pixel_map = get_bad_pixels(dark_list)
        camera.reduction_dict['bad_pixel_mask'].append(~bad_pixel_map)

    if mds_client:
        print("[MDS] Turning source back on...")
        try:
            mds_client.send("on SBB")
            time.sleep(2)
        except Exception as e:
            print(f"[MDS] Warning: {e}")

    print("[CAL] Done with dark acquisition.")

def build_manual_bias(camera, no_frames=100, sleeptime=2):
    print("[CAL] Capturing bias frames...")
    bias_list = camera.get_some_frames(number_of_frames=no_frames, apply_manual_reduction=False)
    bias = np.mean(bias_list, axis=0).astype(int)
    camera.reduction_dict['bias'].append(bias)

def get_bad_pixels(frames, std_threshold=20, mean_threshold=6):
    mean_frame = np.mean(frames, axis=0)
    std_frame = np.std(frames, axis=0)
    global_mean = np.mean(mean_frame)
    global_std = np.std(mean_frame)

    bad_pixel_map = (np.abs(mean_frame - global_mean) > mean_threshold * global_std) | \
                    (std_frame > std_threshold * np.median(std_frame))

    return bad_pixel_map

def build_bad_pixel_mask(camera, bad_pixels, set_bad_pixels_to=0):
    """
    bad_pixels: tuple of (rows, cols) from np.where(condition)
    Updates camera.reduction_dict['bad_pixel_mask'] and also
    flattens the mask for compatibility with 1D operations.
    """
    img_shape = camera.get_image(apply_manual_reduction=False).shape
    bad_pixel_mask = np.ones(img_shape, dtype=int)
    for i, j in zip(*bad_pixels):
        bad_pixel_mask[i, j] = set_bad_pixels_to

    camera.reduction_dict['bad_pixel_mask'].append(bad_pixel_mask)

    flat_mask = np.zeros(img_shape, dtype=bool)
    for i, j in zip(*bad_pixels):
        flat_mask[i, j] = True

    camera.bad_pixel_filter = flat_mask.reshape(-1)
    camera.bad_pixels = np.where(camera.bad_pixel_filter)[0]

def detect_resets(data, threshold=None, axis=(1, 2), min_gap=10, k=15.0):
    """
    Detect reset points in NDRO data by looking for large jumps in mean intensity.
    """
    y = np.mean(data, axis=axis)
    dy = np.abs(np.diff(y))

    if threshold is None:
        mad = np.median(np.abs(dy - np.median(dy)))
        threshold = k * mad
        print(f"[INFO] Auto threshold: {threshold:.2f} (MAD: {mad:.2f})")

    raw_idx = np.where(dy > threshold)[0]
    clean_idx = []

    for i in raw_idx:
        if not clean_idx or (i - clean_idx[-1] >= min_gap):
            clean_idx.append(i)

    return clean_idx

def segment_ndro_stream(data, threshold=15.0):
    """
    Segment 3D NDRO stream (T, H, W) into separate bursts.
    """
    reset_indices = detect_resets(data, threshold=threshold)
    if not reset_indices:
        return [data]  # no reset found

    starts = [0] + [i + 1 for i in reset_indices]
    stops = reset_indices + [data.shape[0]]

    bursts = [data[start:stop] for start, stop in zip(starts, stops)]
    return bursts
