# File: lsc.py
# Description: Lens Shading Correction


import numpy as np

from .basic_module import BasicModule


class LSC(BasicModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        height, width = cfg.hardware.raw_height, cfg.hardware.raw_width
        center_x = width // 2 + 1
        center_y = height // 2 + 1

        x, y = np.meshgrid(np.arange(width), np.arange(height))
        radial_dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        # within flat_radius the lens shading falloff is negligible: clamp the
        # radius there so those pixels all receive the same (minimal) gain
        self.radial_dist = np.where(
            radial_dist <= self.params.flat_radius, self.params.flat_radius, radial_dist
        )

    def execute(self, data):
        bayer = data['bayer'].astype(np.float64)

        gain = self.params.k * self.radial_dist + 1
        lsc_bayer = bayer * gain - self.params.offset
        lsc_bayer = np.clip(lsc_bayer, 0, self.cfg.saturation_values.hdr)

        data['bayer'] = lsc_bayer.astype(np.uint16)
