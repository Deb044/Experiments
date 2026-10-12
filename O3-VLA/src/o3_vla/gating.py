"""A gate persists across control steps, never across independent episodes."""


class Gate:
    def __init__(self, mode, threshold=0.5, smoothing=0.8):
        if mode not in {"base", "always", "hard", "soft", "smoothed", "direct", "oracle"}:
            raise ValueError(mode)
        self.mode, self.threshold, self.smoothing = mode, threshold, smoothing
        self.value = None

    def reset(self):
        self.value = None

    def step(self, probability):
        if not 0 <= probability <= 1:
            raise ValueError("invalid detector probability")
        if self.mode == "base":
            value = 0.0
        elif self.mode == "always":
            value = 1.0
        elif self.mode in {"hard", "direct", "oracle"}:
            value = float(probability >= self.threshold)
        elif self.mode == "soft":
            value = probability
        else:
            value = (
                probability
                if self.value is None
                else (self.smoothing * self.value + (1 - self.smoothing) * probability)
            )
        self.value = value
        return value
