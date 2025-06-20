import time
import numpy as np
from astropy.io import fits
from .shm_reader import SHMReader
from .zmq_utils import CamClient
from . import commands
from .calibration import (
    build_manual_dark,
    build_manual_bias,
    get_bad_pixels,
    build_bad_pixel_mask,
)

class Cred1Camera:
    def __init__(self, shm_path="/dev/shm/cred1.im.shm", semid=3, ip="172.16.8.6", port=6667):
        self.mySHM = SHMReader(shm_path=shm_path, semid=semid)
        self.cam_socket = CamClient(host=ip, port=port)
        self.reduction_dict = {
            'dark': [],
            'bias': [],
            'bad_pixel_mask': []
        }
        self.pupil_crop_region = [None, None, None, None]
        self.bad_pixel_filter = None
        self.bad_pixels = None
        self.config = {}

    def send_fli_cmd(self, command, cmd_sz=10):
        return self.cam_socket.send(command, cmd_sz)

    def print_camera_commands(self):
        print("\n".join(commands.cred1_command_dict.keys()))

    def configure_camera(self, commands_list):
        for cmd in commands_list:
            print(f"[CONFIG] Sending: {cmd}")
            reply = self.send_fli_cmd(cmd)
            print(f"[REPLY] {reply}")
        self.update_config()

    def update_config(self):
        for key, cmd in commands.cred1_command_dict.items():
            try:
                reply = self.send_fli_cmd(cmd)
                self.config[key] = reply.strip()
            except Exception as e:
                print(f"[WARN] Failed to get config '{key}': {e}")

    def get_image(self, apply_manual_reduction=True):
        img = self.mySHM.get_latest()
        if not apply_manual_reduction:
            return img
        return self.apply_manual_reduction(img)

    def get_some_frames(self, number_of_frames=100, apply_manual_reduction=True):
        frame_list = []
        for _ in range(number_of_frames):
            frame = self.mySHM.get_latest()
            if apply_manual_reduction:
                frame = self.apply_manual_reduction(frame)
            frame_list.append(frame)
        return np.array(frame_list)

    def apply_manual_reduction(self, img):
        reduced = img.copy()
        if self.reduction_dict['bias']:
            reduced = reduced - self.reduction_dict['bias'][-1]
        if self.reduction_dict['dark']:
            reduced = reduced - self.reduction_dict['dark'][-1]
        if self.reduction_dict['bad_pixel_mask']:
            mask = self.reduction_dict['bad_pixel_mask'][-1]
            reduced = reduced * mask
        return reduced

    def get_last_raw_image_in_buffer(self):
        return self.mySHM.get_latest_data()

    def get_image_in_another_region(self, region):
        old_region = self.pupil_crop_region.copy()
        self.pupil_crop_region = region
        image = self.get_image()
        self.pupil_crop_region = old_region
        return image

    def save_fits(self, image, fname):
        fits.writeto(fname, image, overwrite=True)

    def close(self):
        self.mySHM.close()
        self.cam_socket.close()


# ─────────────────────────────────────────────────────────────
# Example usage and basic tests
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    def test_single_frame():
        print("[TEST] Acquiring one frame...")
        cam = Cred1Camera()
        img = cam.get_image()
        assert isinstance(img, np.ndarray)
        print(f"Got image of shape {img.shape}")
        cam.close()

    def test_multiple_frames():
        print("[TEST] Acquiring 5 frames...")
        cam = Cred1Camera()
        imgs = cam.get_some_frames(5)
        assert imgs.shape[0] == 5
        print(f"Got {imgs.shape[0]} frames of shape {imgs.shape[1:]}")
        cam.close()

    def test_save_fits():
        print("[TEST] Saving FITS file...")
        cam = Cred1Camera()
        img = cam.get_image()
        fname = "/tmp/test_cred1cam.fits"
        cam.save_fits(img, fname)
        assert os.path.exists(fname)
        print(f"Saved FITS file to {fname}")
        os.remove(fname)
        cam.close()

    def test_config():
        print("[TEST] Updating camera config...")
        cam = Cred1Camera()
        cam.update_config()
        assert "fps" in cam.config or len(cam.config) > 0
        print("Got config keys:", list(cam.config.keys()))
        cam.close()

    # Run all tests
    test_single_frame()
    test_multiple_frames()
    test_save_fits()
    test_config()

