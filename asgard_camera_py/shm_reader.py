import numpy as np
from xaosim.shmlib import shm

class SHMReader:
    def __init__(self, shm_path="/dev/shm/cred1.im.shm", semid=3):
        self.shm_path = shm_path
        self.semid = semid
        self.shm = shm(self.shm_path, nosem=False)

    def get_latest(self):
        """Return the most recent image with semaphore sync (slice-based)."""
        self.shm.catch_up_with_sem(self.semid)
        return self.shm.get_latest_data_slice(self.semid)

    def get_data(self):
        """Return the entire SHM data buffer (for stacked frames)."""
        return self.shm.get_data()

    def get_latest_data(self):
        """Return the most recent frame using get_latest_data."""
        return self.shm.get_latest_data(self.semid)

    def catch_up_with_sem(self):
        """Advance to the latest available frame using semaphore."""
        self.shm.catch_up_with_sem(self.semid)

    def get_shape(self):
        """Return the shape of the SHM buffer or empty tuple on error."""
        try:
            return self.shm.get_data().shape
        except Exception:
            return ()

    def close(self, erase_file=False):
        """Close the shared memory handle, optionally erasing the file."""
        self.shm.close(erase_file=erase_file)

