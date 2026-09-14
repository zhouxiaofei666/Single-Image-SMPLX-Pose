"""User-facing exception hierarchy."""


class ReconstructionError(RuntimeError):
    """Base error for an unsuccessful reconstruction."""


class AssetError(ReconstructionError):
    """Required model code, weights, or licensed assets are missing."""


class DetectionError(ReconstructionError):
    """A usable person crop could not be obtained."""


class InputImageError(ReconstructionError):
    """The supplied image is invalid or unsupported."""
