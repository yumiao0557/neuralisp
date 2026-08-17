# File: nfc.py
# Description: Noise Filtering for Chroma


import numpy as np

from .basic_module import BasicModule, register_dependent_modules
from .helpers import pad, shift_array


@register_dependent_modules('csc')
class NFC(BasicModule):
    def execute(self, data):
        cbcr_image = data['cbcr_image'].astype(np.float64)

        nfc_cbcr_image = np.empty_like(cbcr_image)
        for c in range(cbcr_image.shape[2]):
            channel = cbcr_image[:, :, c]
            padded_channel = pad(channel, pads=1)
            neighbors = np.stack([
                shifted for i, shifted in enumerate(shift_array(padded_channel, window_size=3))
                if i != 4  # exclude the center pixel itself
            ])

            mean = neighbors.mean(axis=0)
            std = neighbors.std(axis=0)
            is_noisy = np.abs(channel - mean) > self.params.thresh * std

            corrected = self.params.alpha * mean + (1 - self.params.alpha) * channel
            nfc_cbcr_image[:, :, c] = np.where(is_noisy, corrected, channel)

        data['cbcr_image'] = np.clip(nfc_cbcr_image, 0, self.cfg.saturation_values.sdr).astype(np.uint8)
