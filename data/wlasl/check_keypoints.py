import json
import math


import json
import numpy as np


def analyze_dataset(file_path):
    """Analyze the dataset for consistent keypoints and NaN presence."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    total_datapoints = len(data)
    datapoints_with_consistent_keypoints = 0
    datapoints_without_nans = 0

    for datapoint in data:
        consistent_keypoints = True
        contains_nans = False

        def has_consistent_keypoints(frames):
            """Check if all frames contain 21 keypoints."""
            return all(len(frame) == 21 for frame in frames)

        def contains_nan(frames):
            """Check if any frame contains NaN values."""
            flat_values = [value for frame in frames for point in frame for value in point]
            array_values = np.array(flat_values, dtype=float)
            return np.isnan(array_values).any()

        # Check for consistency and NaN presence
        left_positions = datapoint['positions']['leftpositions']
        right_positions = datapoint['positions']['rightpositions']

        if not has_consistent_keypoints(left_positions) or not has_consistent_keypoints(right_positions):
            consistent_keypoints = False

        if contains_nan(left_positions) or contains_nan(right_positions):
            contains_nans = True

        # Update counts based on results
        if consistent_keypoints:
            datapoints_with_consistent_keypoints += 1
        if not contains_nans:
            datapoints_without_nans += 1

    # Print results
    print(f'Total datapoints: {total_datapoints}')
    print(
        f'Datapoints with consistent keypoints (21) for both left and right: {datapoints_with_consistent_keypoints}'
    )
    print(
        f'Percentage of datapoints with consistent keypoints: {datapoints_with_consistent_keypoints / total_datapoints * 100:.2f}%'
    )
    print(f'Datapoints without NaNs: {datapoints_without_nans}')
    print(
        f'Percentage of datapoints without NaNs: {datapoints_without_nans / total_datapoints * 100:.2f}%'
    )


# Specify the file paths for the train, validation, and test datasets
# train_file = '/scratch/gpfs/mn4560/final/data/wlasl/main/wlasl_landmark_data_disp_train.json'
# val_file = '/scratch/gpfs/mn4560/final/data/wlasl/main/wlasl_landmark_data_disp_val.json'
# test_file = '/scratch/gpfs/mn4560/final/data/wlasl/main/wlasl_landmark_data_disp_test.json'

train_file = '/scratch/gpfs/mn4560/final/data/asl_citizen/asl_citizen_train_disp.json'
val_file = '/scratch/gpfs/mn4560/final/data/asl_citizen/asl_citizen_val_disp.json'
test_file = '/scratch/gpfs/mn4560/final/data/asl_citizen/asl_citizen_test_disp.json'


print('Analyzing train dataset:')
analyze_dataset(train_file)
print()

print('Analyzing validation dataset:')
analyze_dataset(val_file)
print()

print('Analyzing test dataset:')
analyze_dataset(test_file)
