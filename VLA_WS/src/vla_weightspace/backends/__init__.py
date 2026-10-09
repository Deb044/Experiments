"""Lazy imports keep synthetic verification independent of LeRobot downloads."""


def create_backend(manifest, task, seed):
    if manifest["config"]["backend"] == "synthetic":
        from .synthetic import SyntheticBackend
        return SyntheticBackend(manifest, task, seed)
    from .smolvla import SmolVLABackend
    return SmolVLABackend(manifest, task, seed)
