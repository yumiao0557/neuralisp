# File: bcc.py
# Description: Brightness Contrast Control
# Created: 2021/10/22 20:50
# Author: Qiu Jueqin (qiujueqin@gmail.com)


import numpy as np

from .basic_module import BasicModule, register_dependent_modules


@register_dependent_modules('csc')
class BCC(BasicModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.brightness_offset = np.array(self.params.brightness_offset, dtype=np.int32)
        self.new_max = np.array(self.params.new_max, dtype=np.int32)
        self.new_min = np.array(self.params.new_min, dtype=np.int32)

    def execute(self, data):
        y_image = data['y_image'].astype(np.int32)

        old_max = y_image.max()
        old_min = y_image.min()
        old_mid = (old_max + old_min) / 2
        new_mid = (self.new_max + self.new_min) / 2

        # stretch the frame's own min-max range onto (new_min, new_max)
        contrast_y = (y_image - old_mid) * (self.new_max - self.new_min) / (old_max - old_min) + new_mid

        bcc_y_image = np.clip(contrast_y + self.brightness_offset, 0, 240).astype(np.uint8)

        data['y_image'] = bcc_y_image
