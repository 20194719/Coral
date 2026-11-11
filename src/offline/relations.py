"""Embed mined relation phrases."""

from dataclasses import dataclass
import numpy as np


@dataclass
class RelationCards:
    phrases: list[list[str]]
    embeddings: list[np.ndarray]


def build_cards(cfg, cards_raw):
    """Build cards."""
    phrases = [[card["relation"] for card in cards] for cards in cards_raw]
    flat = [phrase for group in phrases for phrase in group]
    vectors = cfg.encoder(flat)
    embeddings = []
    start = 0
    for group in phrases:
        embeddings.append(vectors[start : start + len(group)])
        start += len(group)
    return RelationCards(phrases, embeddings)
