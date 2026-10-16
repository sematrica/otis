"""Grouping local speakers into global speakers.

Agglomerative clustering with centroid linkage on unit-normalised
embeddings, stopping at a fixed distance threshold, then folding clusters
too small to be a speaker into their nearest large cluster. This is the
algorithm (and the tuned threshold) of pyannote.audio 3.1's
``AgglomerativeClustering`` for the same embedding model, so the number of
speakers is estimated the same way unless the user gives one.

Only "clean" embeddings (at least 2 s of single-speaker speech in the
window) train the clusters; every local speaker is then assigned to the
nearest cluster centroid, and the similarity gap between the best and the
second-best centroid is kept as evidence of how sure that assignment is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: pyannote/speaker-diarization-3.1 tuned values for this embedding model.
THRESHOLD = 0.7045654963945799
MIN_CLUSTER_SIZE = 12
MIN_ACTIVE_RATIO = 0.2
#: Upper bound on embeddings used to build the dendrogram (memory is
#: quadratic in this number). Longer recordings use an even subsample; every
#: embedding is still assigned to the resulting centroids.
MAX_TRAINING_EMBEDDINGS = 6000

UNASSIGNED = -2


@dataclass(frozen=True)
class SpeakerBounds:
    """Optional user knowledge about the number of speakers."""

    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None

    def resolve(self, available: int) -> tuple[int, int, int | None]:
        """(min, max, exact) clamped to what the evidence can support."""
        exact = self.num_speakers
        lo = exact or self.min_speakers or 1
        hi = exact or self.max_speakers or max(1, available)
        lo = max(1, min(available, lo))
        hi = max(1, min(available, hi))
        if lo > hi:
            lo = hi
        return lo, hi, (lo if lo == hi else None)


@dataclass
class ClusteringResult:
    #: (windows, 3) global speaker index per local speaker; UNASSIGNED if none.
    hard: np.ndarray
    #: (windows, 3, K) cosine similarity of each local speaker to each centroid.
    similarity: np.ndarray
    #: (K, 256) cluster centroids (mean of the training embeddings).
    centroids: np.ndarray
    training_count: int
    #: Number of clusters found at the threshold before bounds were applied.
    estimated: int


def training_mask(embeddings: np.ndarray, activity: np.ndarray) -> np.ndarray:
    """(windows, 3) bool: local speakers with enough clean speech to train on."""
    single = activity.sum(axis=2, keepdims=True) == 1
    clean_frames = (activity * single).sum(axis=1)
    active = clean_frames >= MIN_ACTIVE_RATIO * activity.shape[1]
    valid = ~np.isnan(embeddings).any(axis=2)
    return active & valid


def _normalise(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def agglomerative(train: np.ndarray, min_clusters: int, max_clusters: int, num_clusters: int | None) -> tuple[np.ndarray, int]:
    """Cluster labels (0..K-1) for the training embeddings, and the number of
    large clusters found at the threshold (the unconstrained estimate)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import cdist

    n = train.shape[0]
    if n == 1:
        return np.zeros(1, dtype=np.int64), 1
    min_size = min(MIN_CLUSTER_SIZE, max(1, round(0.1 * n)))
    unit = _normalise(train.astype(np.float64))
    dendrogram = linkage(unit, method="centroid", metric="euclidean")
    clusters = fcluster(dendrogram, THRESHOLD, criterion="distance") - 1
    unique, counts = np.unique(clusters, return_counts=True)
    estimated = int(np.sum(counts >= min_size))

    target = None
    if estimated < min_clusters:
        target = min_clusters
    elif estimated > max_clusters:
        target = max_clusters
    if num_clusters is not None:
        target = num_clusters
    if target is not None and target != estimated:
        # Walk the dendrogram outwards from the threshold until the number of
        # large clusters matches the requested number (pyannote's approach).
        by_iteration = np.copy(dendrogram)
        by_iteration[:, 2] = np.arange(n - 1)
        best_iteration, best_large = n - 1, 1
        for iteration in np.argsort(np.abs(dendrogram[:, 2] - THRESHOLD)):
            if by_iteration[iteration, 3] < min_size:
                continue
            candidate = fcluster(by_iteration, iteration, criterion="distance") - 1
            _, candidate_counts = np.unique(candidate, return_counts=True)
            large = int(np.sum(candidate_counts >= min_size))
            if abs(large - target) < abs(best_large - target):
                best_iteration, best_large = iteration, large
            if large == target:
                break
        clusters = fcluster(by_iteration, best_iteration, criterion="distance") - 1
        unique, counts = np.unique(clusters, return_counts=True)

    large_ids = unique[counts >= min_size]
    if large_ids.size == 0:
        return np.zeros(n, dtype=np.int64), estimated
    small_ids = unique[counts < min_size]
    if small_ids.size:
        large_centroids = np.vstack([unit[clusters == k].mean(axis=0) for k in large_ids])
        small_centroids = np.vstack([unit[clusters == k].mean(axis=0) for k in small_ids])
        nearest = np.argmin(cdist(large_centroids, small_centroids, metric="cosine"), axis=0)
        for small, large in zip(small_ids, nearest):
            clusters[clusters == small] = large_ids[large]
    _, labels = np.unique(clusters, return_inverse=True)
    return labels.astype(np.int64), estimated


def cluster(
    embeddings: np.ndarray,
    activity: np.ndarray,
    bounds: SpeakerBounds = SpeakerBounds(),
    *,
    constrained: bool = False,
) -> ClusteringResult:
    """Assign every local speaker of every window to a global speaker.

    ``embeddings``: (windows, 3, 256) with NaN where none could be computed.
    ``activity``: (windows, frames, 3) binary local-speaker activity.
    """
    from scipy.optimize import linear_sum_assignment

    windows, local, _ = embeddings.shape
    mask = training_mask(embeddings, activity)
    if not mask.any():
        # Short recordings or fragmented speech: train on whatever exists.
        mask = ~np.isnan(embeddings).any(axis=2)
    hard = np.full((windows, local), UNASSIGNED, dtype=np.int64)
    if not mask.any():
        return ClusteringResult(hard, np.zeros((windows, local, 0)), np.zeros((0, embeddings.shape[2])), 0, 0)

    chunk_idx, speaker_idx = np.nonzero(mask)
    if chunk_idx.size > MAX_TRAINING_EMBEDDINGS:
        keep = np.linspace(0, chunk_idx.size - 1, MAX_TRAINING_EMBEDDINGS).round().astype(np.int64)
        chunk_idx, speaker_idx = chunk_idx[keep], speaker_idx[keep]
    train = embeddings[chunk_idx, speaker_idx]
    lo, hi, exact = bounds.resolve(train.shape[0])
    if hi < 2:
        labels, estimated = np.zeros(train.shape[0], dtype=np.int64), 1
    else:
        labels, estimated = agglomerative(train, lo, hi, exact)

    k = int(labels.max()) + 1
    centroids = np.vstack([train[labels == i].mean(axis=0) for i in range(k)])
    flat = embeddings.reshape(-1, embeddings.shape[2])
    valid = ~np.isnan(flat).any(axis=1)
    similarity = np.full((flat.shape[0], k), np.nan)
    similarity[valid] = _normalise(flat[valid].astype(np.float64)) @ _normalise(centroids.astype(np.float64)).T
    similarity = similarity.reshape(windows, local, k)

    has_embedding = ~np.isnan(similarity).any(axis=2)
    if constrained:
        for c in range(windows):
            rows = np.nonzero(has_embedding[c])[0]
            if rows.size == 0:
                continue
            r, cols = linear_sum_assignment(similarity[c, rows], maximize=True)
            hard[c, rows[r]] = cols
    else:
        best = np.argmax(np.nan_to_num(similarity, nan=-np.inf), axis=2)
        hard[has_embedding] = best[has_embedding]
    return ClusteringResult(hard, similarity, centroids, int(train.shape[0]), estimated)
