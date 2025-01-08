from collections.abc import Set
from typing import Optional
import numpy as np
import pandas as pd
from configs import FILES_DIRECTORY
from query_parse.visual import clip_model, siglip_model
from tqdm.auto import tqdm

photo_ids = siglip_model.photo_ids

blurred = pd.read_csv(f"/home/allie/highres/LSC23/blurred.csv")


def to_key(image):
    return "/".join(image.split("/")[-3:])


blurred["laplacian_var"] = blurred["laplacian_var"].fillna(0)
blurred_images = blurred[blurred["laplacian_var"] < 100]["image"].apply(to_key).tolist()
blurred_images = set(blurred_images)
blurred_indices = [i for i, image in enumerate(photo_ids) if image in blurred_images]
blurred_indices = set(blurred_indices)


def detect_noise(scores1, scores2, method="absolute", threshold=None, percentile=95):
    """
    Detect noisy indices based on the difference between two score arrays.

    Parameters:
    - scores1: First array of scores.
    - scores2: Second array of scores.
    - method: "absolute" or "relative".
    - threshold: Fixed threshold for noise detection. If None, use percentile.
    - percentile: Percentile to dynamically compute the threshold if not provided.

    Returns:
    - noisy_indices: Indices where the difference exceeds the threshold.
    - differences: Array of differences.
    """
    # Compute differences
    if method == "absolute":
        differences = np.abs(scores1 - scores2)
    elif method == "relative":
        differences = np.abs(scores1 - scores2) / (
            np.abs(scores1) + 1e-8
        )  # Avoid division by zero
    else:
        raise ValueError(f"Unknown method: {method}")

    # Determine the threshold
    if threshold is None:
        threshold = np.percentile(differences, percentile)

    # Detect noisy indices
    noisy_indices = np.where(differences > threshold)[0]

    return noisy_indices, differences


def estimate_variance_threshold(
    normalized_scores, method="global", scaling_factor=1.5, sample_size=1000
):
    """
    Estimate the variance threshold without computing all possible segment variances.

    Parameters:
    - normalized_scores: Array of normalized scores.
    - method: "global", "sampling", "statistics", "percentile".
    - scaling_factor: Multiplier for variance-based thresholds.
    - sample_size: Number of samples to use for "sampling" method.

    Returns:
    - Estimated variance threshold.
    """
    if method == "global":
        # Use global variance
        global_variance = np.var(normalized_scores)
        return global_variance * scaling_factor

    elif method == "sampling":
        # Randomly sample variances from limited segments
        sampled_indices = np.random.choice(
            len(normalized_scores), sample_size, replace=False
        )
        sampled_variances = [
            np.var(normalized_scores[i : i + 10])
            for i in sampled_indices
            if i + 10 < len(normalized_scores)
        ]
        return np.percentile(sampled_variances, 75)  # Use the 75th percentile

    elif method == "statistics":
        # Estimate based on mean and standard deviation
        std_normalized_scores = np.std(normalized_scores)
        return (std_normalized_scores**2) * scaling_factor

    elif method == "percentile":
        # Estimate based on score distribution percentiles
        spread = np.percentile(normalized_scores, 90) - np.percentile(
            normalized_scores, 10
        )
        return (spread**2) / scaling_factor

    else:
        raise ValueError(f"Unknown method: {method}")


blurred = pd.read_csv("/home/allie/highres/LSC23/blurred.csv")


def get_siglip_model_scores(query):
    encoded_query = siglip_model.encode_text(query)
    scores = siglip_model.norm_photo_features @ encoded_query.T
    return scores, encoded_query


def get_clip_scores(query):
    encoded_query = clip_model.encode_text(query)
    scores = clip_model.norm_photo_features @ encoded_query.T
    return scores, encoded_query


lambda1 = 0.75
lambda2 = 1 - lambda1


def get_group_score(segment, encoded_query, clip_encoded_query):
    siglip_model_score = siglip_model.norm_photo_features[segment] @ encoded_query.T
    clip_score = clip_model.norm_photo_features[segment] @ clip_encoded_query.T
    return np.mean(lambda1 * siglip_model_score + lambda2 * clip_score)


def get_segments(
    query: str,
    score_percentile: int = 90,
    variance_percentile: int = 75,
    max_gap: int = 2,
    max_length: int = 100,
    filters: Optional[Set[str]] = None,
):
    scores, encoded_query = get_siglip_model_scores(query)
    clip_scores, clip_encoded_query = get_clip_scores(query)
    scores = lambda1 * scores + lambda2 * clip_scores

    # Filter indices based on filters
    if filters is None:
        filter_indices = set(range(len(photo_ids)))
    else:
        filter_indices = set(
            i for i, image in enumerate(photo_ids) if image in filters
        )

    # Dynamically determine the score threshold
    score_threshold = np.percentile(scores, score_percentile)
    print(f"Dynamic score threshold (percentile {score_percentile}): {score_threshold}")

    # Filter high-scoring images
    high_score_indices = np.where(scores >= score_threshold)[0]
    # Filter out blurred images
    high_score_indices = [
        idx for idx in high_score_indices if idx not in blurred_indices
    ]
    # Keep only filtered indices
    high_score_indices = [
        idx for idx in high_score_indices if idx in filter_indices
    ]
    if len(high_score_indices) == 0:
        print("No high-scoring images found.")
        return [], []

    # Normalize scores for segment scoring
    max_score = np.max(scores)
    normalized_scores = scores / max_score

    # Dynamically determine the variance threshold
    variance_threshold = estimate_variance_threshold(
        normalized_scores=normalized_scores, method="sampling", sample_size=1000
    )
    print(
        f"Dynamic variance threshold (percentile {variance_percentile}): {variance_threshold}"
    )

    def expand_segment(seed, direction):
        """
        Expand the segment starting from a seed index in the given direction.
        - direction: 1 for right, -1 for left
        """
        segment = [seed]
        gap_count = 0
        for step in range(1, max_length):
            next_idx = seed + direction * step
            if next_idx < 0 or next_idx >= len(scores):
                break  # Out of bounds

            if scores[next_idx] >= score_threshold or gap_count < max_gap:
                segment.append(next_idx)
                if (
                    scores[next_idx] < score_threshold
                    or next_idx in blurred_indices
                ):
                    gap_count += 1
            else:
                break  # Stop expanding if variance exceeds threshold or max gap is reached

            # Check variance
            segment_scores = normalized_scores[segment]
            if np.var(segment_scores) > variance_threshold:
                segment.pop()  # Remove the last added index
                break

        return sorted(segment)  # Ensure the segment is in order

    segments = []
    used_indices = set()

    for seed in tqdm(high_score_indices):
        if seed in used_indices:
            continue  # Skip seeds already covered in a segment

        # Expand left and right
        left_segment = expand_segment(seed, direction=-1)
        right_segment = expand_segment(seed, direction=1)
        full_segment = sorted(set(left_segment + right_segment))

        # Calculate segment score
        segment_scores = normalized_scores[full_segment]
        segment_length = len(full_segment)
        group_score = (
            get_group_score(full_segment, encoded_query, clip_encoded_query) / max_score
        )
        variance = np.var(segment_scores)
        if segment_length == 1:
            length_reward = 0.0
        elif segment_length >= 50:
            length_reward = 0.05
        else:
            length_reward = 0.05 * np.log(segment_length)  # Example length reward

        segment_score = group_score - variance + length_reward

        # Mark indices as used
        used_indices.update(full_segment)
        segments.append(
            (full_segment, segment_score, group_score, variance, length_reward)
        )

    # Sort segments by score
    segments.sort(key=lambda x: x[1], reverse=True)

    # Convert segment indices back to ranges
    result_segments = []
    result_scores = []
    i = 0
    for segment, seg_score, group_score, variance, length_reward in segments:
        start, end = segment[0], segment[-1]
        result_segments.append((start, end + 1))  # Include end index
        result_scores.append(seg_score)
        i += 1
        if i <= 10:
            print(f"Segment {i}: Start: {start}, End: {end}, Length: {end - start + 1}")
            print(
                f"Segment score: {seg_score:0.2f}, Group score: {group_score:0.2f}, Variance: {variance:0.2f}, Length reward: {length_reward:0.2f}"
            )

    return result_segments[:500], result_scores[:500]


# query = "I am running on a treadmill"
# segments, scores = get_segments(query=query, max_gap=5)
# print("Number of segments", len(segments))
