from collections.abc import Set
from typing import Optional

import numpy as np
import pandas as pd
from configs import CLIP_EMBEDDINGS
from database.main import get_db, image_collection
from query_parse.types.requests import Data
from query_parse.visual import clip_model, clipa_model, get_model, photo_ids
from tqdm.auto import tqdm


def get_blurred_indices(data: Data):
    blurred_df = pd.read_csv(f"{CLIP_EMBEDDINGS}/{data}/blurred.csv")
    blurred_df["laplacian_var"] = blurred_df["laplacian_var"].fillna(0)
    blurred_images = blurred_df[blurred_df["laplacian_var"] < 100]["image"].tolist()
    blurred_images = set(blurred_images)
    blurred_indices = [
        i for i, image in enumerate(photo_ids(data)) if image in blurred_images
    ]
    return set(blurred_indices)


blurred_indices = {
    Data.LSC23: get_blurred_indices(Data.LSC23),
    Data.Deakin: get_blurred_indices(Data.Deakin),
}


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


lambda1 = 0.75
lambda2 = 1 - lambda1


def compare_images(data: Data, image1: dict, image2: dict):
    """
    For the LSC23 dataset, fixed boundaries are:
    - different locations
    - time gap where the camera is off
    For Deakin, fixed boundaries are:
    - different patientID
    - time gap where the camera is off
    """
    time_gap = 60 * 5  # 5 minutes
    if data == Data.LSC23:
        if image1["location"] != image2["location"]:
            return True
        if (image1["utc_time"] - image2["utc_time"]).total_seconds() > time_gap:
            return True
    elif data == Data.Deakin:
        if image1["patient"]["id"] != image2["patient"]["id"]:
            return True
        if (
            image1["snap"]["local_time"] - image2["snap"]["local_time"]
        ).total_seconds() > time_gap:
            return True
        if image1["date"] != image2["date"]:
            return True

    return False


def get_fixed_boundaries(data):
    sort_criteria = None
    if data == Data.LSC23:
        sort_criteria = [("utc_time", 1)]
    elif data == Data.Deakin:
        sort_criteria = [("patient.id", 1), ("snap.local_time", 1)]

    boundaries = set()
    prev_image = None
    for image_data in image_collection(get_db(data)).find().sort(sort_criteria):
        if prev_image is not None:
            if compare_images(data, prev_image, image_data):
                try:
                    boundaries.add(photo_ids(data).index(image_data["image"]))
                except ValueError:
                    pass
        prev_image = image_data
    return boundaries


FIXED_BOUNDARIES = {
    Data.LSC23: get_fixed_boundaries(Data.LSC23),
    Data.Deakin: get_fixed_boundaries(Data.Deakin),
}


def get_segments(
    query: str,
    data: Data = Data.LSC23,
    score_percentile: int = 95,
    variance_percentile: int = 75,
    max_gap: int = 2,
    max_length: int = 100,
    filters: Optional[Set[str]] = None,
    to_merge: bool = False
):
    chosen_model = get_model(data)
    alternative_model = clip_model if data == Data.LSC23 else clipa_model

    scores, encoded_query = chosen_model.score_all_images(query, data)
    photo_ids = chosen_model.photo_ids[data]

    alternative_scores, alternative_encoded_query = alternative_model.score_all_images(
        query, data
    )
    scores = [lambda1 * s + lambda2 * c for s, c in zip(scores, alternative_scores)]

    def get_group_score(segment):
        chosen_model_score = (
            chosen_model.norm_photo_features[data][segment] @ encoded_query.T
        )
        alternative_score = (
            alternative_model.norm_photo_features[data][segment]
            @ alternative_encoded_query.T
        )
        return np.mean(lambda1 * chosen_model_score + lambda2 * alternative_score)

    # Filter out blurred images
    good_indices = [
        i for i, image in enumerate(photo_ids) if image not in blurred_indices
    ]
    # Filter out images that do not match the filters
    filter_indices = set(range(len(photo_ids)))
    if filters:
        filter_indices = set(i for i, image in enumerate(photo_ids) if image in filters)

    # Dynamically determine the score threshold
    good_indices = [
        idx for idx in good_indices if idx in filter_indices
    ]
    okay_scores = [scores[i] for i in good_indices]
    score_threshold = np.percentile(okay_scores, score_percentile)
    print(f"Dynamic score threshold (percentile {score_percentile}): {score_threshold}")

    # Filter high-scoring images
    high_score_indices = np.where(scores > score_threshold)[0]
    good_indices = [idx for idx in good_indices if idx in high_score_indices]

    # Keep only filtered indices
    print(f"Found {len(good_indices)} high-scoring images.")
    if len(good_indices) == 0:
        print("No high-scoring images found.")
        return {
            "segments": [],
            "segment_scores": [],
            "scores": [],
            "high_score_indices": [],
        }

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

            if direction == 1 and next_idx in FIXED_BOUNDARIES[data]:
                # Stop expanding if fixed boundary is reached
                break

            if direction == -1 and next_idx + 1 in FIXED_BOUNDARIES[data]:
                # Using next_idx + 1 to get the right boundary
                break

            if next_idx < 0 or next_idx >= len(scores):
                break  # Out of bounds

            if scores[next_idx] >= score_threshold or gap_count < max_gap:
                segment.append(next_idx)
                if (
                    scores[next_idx] < score_threshold
                    or next_idx in blurred_indices[data]
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

    hits = set(good_indices)
    for seed in tqdm(good_indices):
        if seed in used_indices:
            continue  # Skip seeds already covered in a segment

        # Expand left and right
        left_segment = expand_segment(seed, direction=-1)
        right_segment = expand_segment(seed, direction=1)
        full_segment = sorted(set(left_segment + right_segment))

        if len(full_segment) == 0:
            continue

        # Add segment to hits
        hits.update(full_segment)

        # Calculate segment score
        segment_scores = normalized_scores[full_segment]
        segment_length = len(full_segment)
        group_score = get_group_score(full_segment) / max_score
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

    # Merge overlapping segments
    if not to_merge:
        merged_segments = segments
    else:
        # sorted segments by start index
        segments.sort(key=lambda x: x[0][0])
        merged_segments = []
        i = 0
        while i < len(segments):
            segment = segments[i]
            start, end = segment[0][0], segment[0][-1]
            seg_score, group_score, variance, length_reward = (
                segment[1],
                segment[2],
                segment[3],
                segment[4],
            )
            j = i + 1
            while j < len(segments):
                next_segment = segments[j]
                next_start, next_end = next_segment[0][0], next_segment[0][-1]
                next_seg_score, _, next_variance, next_length_reward = (
                    next_segment[1],
                    next_segment[2],
                    next_segment[3],
                    next_segment[4],
                )
                if next_start <= end:
                    # Overlapping segments
                    end = max(end, next_end)
                    seg_score += next_seg_score
                    group_score = get_group_score(range(start, end + 1)) / max_score
                    variance += next_variance
                    length_reward += next_length_reward
                    j += 1
                else:
                    break

            merged_segments.append(
                ([start, end], seg_score, group_score, variance, length_reward)
            )
            i = j


    # Sort segments by score
    merged_segments.sort(key=lambda x: x[1], reverse=True)

    # Convert segment indices back to ranges
    result_segments = []
    result_scores = []
    i = 0
    for segment, seg_score, group_score, variance, length_reward in merged_segments:
        start, end = segment[0], segment[-1]
        result_segments.append((start, end + 1))  # Include end index
        result_scores.append(seg_score)
        i += 1
        if i <= 10:
            print(f"Segment {i}: Start: {start}, End: {end}, Length: {end - start + 1}")
            print(
                f"Segment score: {seg_score:0.2f}, Group score: {group_score:0.2f}, Variance: {variance:0.2f}, Length reward: {length_reward:0.2f}"
            )

    return {
        "segments": result_segments,
        "segment_scores": result_scores,
        "scores": scores,
        "high_score_indices": hits,
    }


# query = "I am having food or interacting with food"
# results = get_segments(query=query, data=Data.Deakin, max_gap=5)
# segments = results["segments"]
# print("Number of segments", len(segments))
# images = []
# deakin_photo_ids = photo_ids(Data.Deakin)
# for start, end in segments:
#     images.append(deakin_photo_ids[start:end])
# with open("segments.txt", "w") as f:
#     for image_list in images:
#         f.write(", ".join(image_list))
#         f.write("\n")
