"""Inference, and an honest confidence.

The confidence returned here is the mean per-step probability of the characters
actually emitted, not a softmax peak. It is still NOT a calibrated threshold --
scripts/eval_plates.py measures where it should sit against the degradation
ladder, because the V-C3 probe already showed a general OCR reading correctly at
0.74, which is precisely the trap of treating a raw score as a threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .dataset import to_tensor
from .model import BLANK, CHARS, PlateNet

DEFAULT_WEIGHTS = Path("models/plate_crnn.pt")


@dataclass(frozen=True, slots=True)
class CharacterRead:
    """One read, with the per-character confidences the mean was taken over.

    These numbers were always computed -- the confidence `read()` returns IS
    their mean -- and were discarded on the way out. They are what a later stage
    needs to say WHICH character it is unsure of, rather than only that the read
    as a whole is uncertain.

    It is a separate accessor rather than a third return value, and that is a
    measured decision, not a stylistic one. `read()` returning `(text, conf)` is
    a PUBLISHED contract: `PlateEngine`'s `recognizer=` is a documented
    injection point whose stated shape is "anything with
    `.read(image) -> (text, conf)`", and third-party recognisers are written to
    it. Widening the tuple breaks `engine.py`, six call sites in
    `scripts/eval_plates.py`, both test stubs and every such recogniser, in
    return for information no caller has yet asked for.
    """

    text: str
    confidence: float
    per_character: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.per_character) != len(self.text):
            raise ValueError(
                f"{len(self.per_character)} confidences for {len(self.text)} "
                "characters; they are per emitted character, one each."
            )


class PlateRecognizer:
    def __init__(self, weights: Path = DEFAULT_WEIGHTS, device: str = "cpu") -> None:
        if not Path(weights).exists():
            raise FileNotFoundError(
                f"no weights at {weights}. They are not committed by design -- "
                "rebuild them with `python -m vehicle_id.plates.train`."
            )
        self.device = torch.device(device)
        self.model = PlateNet().to(self.device)
        blob = torch.load(weights, map_location=self.device)
        self.model.load_state_dict(blob["state_dict"])
        self.model.eval()

    def read(self, image: np.ndarray) -> tuple[str, float]:
        """The published shape: `(text, confidence)`, and nothing else.

        Unchanged, and it stays unchanged. This is the tuple `engine.py`, the
        plate harness and any third-party recogniser are written against.
        """
        result = self.read_characters(image)
        return result.text, result.confidence

    @torch.no_grad()
    def read_characters(self, image: np.ndarray) -> CharacterRead:
        """The same read, plus the per-character confidences it already computed.

        The mean of `per_character` IS `confidence` -- one number derived from
        the other, in one place, so the two cannot drift into disagreeing about
        the same read.
        """
        x = to_tensor(image).unsqueeze(0).to(self.device)
        probs = self.model(x).softmax(2)[0]           # T, C
        best = probs.argmax(1)

        text, kept, previous = [], [], BLANK
        for t, index in enumerate(best.tolist()):
            if index != previous and index != BLANK:
                text.append(CHARS[index - 1])
                kept.append(probs[t, index].item())
            previous = index

        if not text:
            return CharacterRead(text="", confidence=0.0, per_character=())
        return CharacterRead(
            text="".join(text),
            confidence=float(np.mean(kept)),
            per_character=tuple(float(k) for k in kept),
        )
