import torch
import torch.nn as nn

class ModuleAttrMixin(nn.Module):
    def __init__(self):
        super().__init__()
        # Register a dummy parameter to ensure device/dtype detection works
        # even for modules without trainable parameters
        self.register_buffer('_device_buffer', torch.tensor(0.0))

    @property
    def device(self):
        # Try to get device from parameters first
        try:
            return next(iter(self.parameters())).device
        except StopIteration:
            # Fallback to buffer if no parameters
            return self._device_buffer.device
    
    @property
    def dtype(self):
        # Try to get dtype from parameters first
        try:
            return next(iter(self.parameters())).dtype
        except StopIteration:
            # Fallback to buffer if no parameters
            return self._device_buffer.dtype
