# File: hsc.py
# Description: Hue Saturation Control
# Created: 2021/10/22 20:50
# Author: Qiu Jueqin (qiujueqin@gmail.com)


import numpy as np

from .basic_module import BasicModule, register_dependent_modules


@register_dependent_modules('csc')
class HSC(BasicModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.delta_theta = np.pi * self.params.hue_offset / 180
        self.s = np.array(self.params.saturation_intensity, dtype=np.float64)

    def execute(self, data):
        cbcr_image = data['cbcr_image'].astype(np.uint16)

        cb = cbcr_image[:, :, 0].astype(np.float64) + 1  # +1 avoids the (0, 0) singularity in arctan2
        cr = cbcr_image[:, :, 1].astype(np.float64) + 1

        theta = np.arctan2(cr, cb)
        dist = np.sqrt(cb ** 2 + cr ** 2)

        hsc_cb = self.s * dist * np.cos(theta + self.delta_theta)
        hsc_cr = self.s * dist * np.sin(theta + self.delta_theta)

        hsc_cbcr_image = np.empty_like(cbcr_image, dtype=np.uint8)
        hsc_cbcr_image[:, :, 0] = np.clip(hsc_cb, 0, self.cfg.saturation_values.sdr)
        hsc_cbcr_image[:, :, 1] = np.clip(hsc_cr, 0, self.cfg.saturation_values.sdr)

        data['cbcr_image'] = hsc_cbcr_image
