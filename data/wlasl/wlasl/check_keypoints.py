import json
import math


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

        # Check for consistent keypoints (21) in left and right positions
        for frame in datapoint['positions']['leftpositions']:
            if len(frame) != 21:
                consistent_keypoints = False
                break
            if any(math.isnan(value) for point in frame for value in point):
                contains_nans = True

        for frame in datapoint['positions']['rightpositions']:
            if len(frame) != 21:
                consistent_keypoints = False
                break
            if any(math.isnan(value) for point in frame for value in point):
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
train_file = '/Users/nguyen/Desktop/princeton/cos/cos429/final/data/wlasl/wlasl_train_centered.json'
val_file = '/Users/nguyen/Desktop/princeton/cos/cos429/final/data/wlasl/wlasl_val_centered.json'
test_file = '/Users/nguyen/Desktop/princeton/cos/cos429/final/data/wlasl/wlasl_test_centered.json'

print('Analyzing train dataset:')
analyze_dataset(train_file)
print()

print('Analyzing validation dataset:')
analyze_dataset(val_file)
print()

print('Analyzing test dataset:')
analyze_dataset(test_file)
