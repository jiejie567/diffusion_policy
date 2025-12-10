from typing import Dict
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

class BaseImageRunner:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def run(self, policy: BaseImagePolicy) -> Dict:
        raise NotImplementedError()


class NoOpImageRunner(BaseImageRunner):
    """
    Dummy runner for offline-only training; accepts extra kwargs and returns empty log.
    """
    def __init__(self, output_dir=None, **kwargs):
        super().__init__(output_dir=output_dir)

    def run(self, policy: BaseImagePolicy) -> Dict:
        return {}
