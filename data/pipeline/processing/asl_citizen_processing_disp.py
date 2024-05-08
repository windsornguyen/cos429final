#################
# IMPORTS
#################

import cv2
import mediapipe as mp
import os
import copy
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import multiprocessing
import logging
from tqdm import tqdm
import pickle
import csv

# Set up logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)

#####################
# Extract features from ASL videos via keypoints for LHS and RHS.
# We remove noise and inaccuracies in the keypoint models by only storing data
# from every other frame. This gives us a time series for each video where we know the
# (1) number of hands in the frame, (2) keypoints for the left hand, (3)
# keypoints for the right hand
######################


def process_video(input_file):
    try:
        # Initialize the Mediapipe Hands solution
        mp_hands = mp.solutions.hands
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
        ) as hands:
            # Open the video file
            cap = cv2.VideoCapture(input_file)
            if not cap.isOpened():
                logging.error(f'Failed to open video file: {input_file}')
                return None

            # Initialize lists to store landmark data
            left_hand_vectors, right_hand_vectors, num_hands_list = [], [], []
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % 2 == 0:  # Process every other frame for efficiency
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(frame_rgb)
                    num_hands = (
                        len(results.multi_hand_landmarks)
                        if results.multi_hand_landmarks
                        else 0
                    )
                    num_hands_list.append(num_hands)

                    left_hand_landmarks = [(-1, -1)] * 21
                    right_hand_landmarks = [(-1, -1)] * 21

                    if results.multi_hand_landmarks:
                        for hand_landmarks, handedness in zip(
                            results.multi_hand_landmarks,
                            results.multi_handedness,
                        ):
                            hand_type = handedness.classification[0].label
                            landmarks = [
                                (landmark.x, landmark.y)
                                for landmark in hand_landmarks.landmark
                            ]

                            if hand_type == 'Left':
                                left_hand_landmarks = landmarks
                            else:
                                right_hand_landmarks = landmarks

                    left_hand_vectors.append(left_hand_landmarks)
                    right_hand_vectors.append(right_hand_landmarks)

                frame_count += 1

            # Release the video capture object
            cap.release()

            # Return the landmark data
            return {
                'filename': os.path.basename(input_file),
                'numHands': num_hands_list,
                'left_vectors': left_hand_vectors,
                'right_vectors': right_hand_vectors,
            }

    except Exception as e:
        logging.error(f'Error processing video {input_file}: {str(e)}')
        return None


def process_videos_in_folder(folder_path, checkpoint_file):
    video_files = [
        filename for filename in os.listdir(folder_path) if filename.endswith('.mp4')
    ]

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            features_dict = pickle.load(f)
            processed_files = set(features_dict.keys())
    else:
        features_dict = {}
        processed_files = set()

    remaining_files = [
        filename for filename in video_files if filename not in processed_files
    ]

    with multiprocessing.Pool(processes=8) as pool:
        results = list(
            tqdm(
                pool.imap(
                    process_video,
                    [
                        os.path.join(folder_path, filename)
                        for filename in remaining_files
                    ],
                ),
                total=len(remaining_files),
            )
        )

    features_dict.update(
        {result['filename']: result for result in results if result is not None}
    )

    with open(checkpoint_file, 'wb') as f:
        pickle.dump(features_dict, f)

    return features_dict


def min_max_scale(positions):
    """Scale the positions data between 0 and 1, handling all-NaN cases."""
    positions_array = np.array(
        positions, dtype=float
    )  # Ensure correct type for NaN handling
    positions_array[positions_array == -1] = np.nan  # Replace placeholder -1 with NaN

    def safe_nanmin(array):
        return np.nanmin(array) if not np.all(np.isnan(array)) else None

    def safe_nanmax(array):
        return np.nanmax(array) if not np.all(np.isnan(array)) else None

    # Calculate min and max values safely
    min_x = safe_nanmin(positions_array[:, 0])
    max_x = safe_nanmax(positions_array[:, 0])
    min_y = safe_nanmin(positions_array[:, 1])
    max_y = safe_nanmax(positions_array[:, 1])

    if min_x is None or max_x is None or min_y is None or max_y is None:
        logging.warning('All-NaN slice encountered while computing min/max values.')
        positions_array[np.isnan(positions_array)] = -1  # Replace NaN back to -1
        return positions_array.tolist()

    # Scale non-NaN values to the range [0, 1]
    positions_array[:, 0] = (positions_array[:, 0] - min_x) / (max_x - min_x)
    positions_array[:, 1] = (positions_array[:, 1] - min_y) / (max_y - min_y)

    # Replace NaN back to placeholder -1
    positions_array[np.isnan(positions_array)] = -1

    return positions_array.tolist()


def calculate_displacements(positions):
    displacements = []
    for i in range(len(positions) - 1):
        current_vector = positions[i]
        next_vector = positions[i + 1]

        displacement = []
        for j in range(21):
            if current_vector[j] == (-1, -1) or next_vector[j] == (-1, -1):
                displacement.append((-1, -1))
            else:
                displacement.append(
                    (
                        next_vector[j][0] - current_vector[j][0],
                        next_vector[j][1] - current_vector[j][1],
                    )
                )

        displacements.append(displacement)

    return displacements


def zero_mean_center(displacements):
    displacements_array = np.array(displacements)
    displacements_array = displacements_array.astype(
        float
    )  # Convert to float to allow NaN
    displacements_array[displacements_array == -1] = np.nan  # Replace -1 with NaN

    # Compute mean displacement excluding NaN values
    mean_displacement = np.nanmean(displacements_array)

    # Center the displacements by subtracting the mean
    centered_displacements = displacements_array - mean_displacement

    # Replace NaN back to -1
    centered_displacements[np.isnan(centered_displacements)] = -1

    return centered_displacements.tolist()


##############################################
# general set creation helpers
##############################################

def remove_trailing_numbers(s):
    while s and s[-1].isdigit():
        s = s[:-1]
    return s

def create_video_to_gloss_dict(file_path):
    video_to_gloss = {}

    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            video_file = row['Video file']
            gloss = row['Gloss']
            gloss = remove_trailing_numbers(gloss)
            video_to_gloss[video_file] = gloss

    return video_to_gloss


if __name__ == '__main__':
    video_folder = '/scratch/gpfs/mn4560/final/data/asl_citizen/videos'
    checkpoint_file = '/scratch/gpfs/mn4560/final/data/asl_citizen/asl_landmark_data_centered_chkpt.pkl'
    landmark_data = process_videos_in_folder(video_folder, checkpoint_file)

    ######################################
    # Observe that each video has extraneous frames where we have no
    # hands present at the start and end. We now clean the data by removing
    # these so each time-series only contains the data we actually care about
    ######################################

    # Assuming landmark_data is defined
    for video_id in landmark_data:
        li = landmark_data[video_id]['numHands'].copy()
        first_non_zero_index = next((i for i, x in enumerate(li) if x != 0), 0)
        last_non_zero_index = next(
            (i for i, x in enumerate(reversed(li)) if x != 0), len(li) - 1
        )

        # Cleaning rhs and lhs vectors based on first and last non_zero entry
        start = first_non_zero_index
        end = len(li) - last_non_zero_index - 1
        landmark_data[video_id]['start'] = start
        landmark_data[video_id]['end'] = end
        landmark_data[video_id]['cleaned_left_vectors'] = landmark_data[video_id][
            'left_vectors'
        ][start : end + 1]
        landmark_data[video_id]['cleaned_right_vectors'] = landmark_data[video_id][
            'right_vectors'
        ][start : end + 1]

    # Copy everything so we have correct data
    cleaned_data = copy.deepcopy(
        landmark_data
    )  # Use deepcopy to ensure full independence
    for video_id in cleaned_data:
        cleaned_data[video_id].pop('numHands')
        cleaned_data[video_id].pop('start')
        cleaned_data[video_id].pop('end')
        cleaned_data[video_id]['leftpositions'] = cleaned_data[video_id][
            'cleaned_left_vectors'
        ]
        cleaned_data[video_id]['rightpositions'] = cleaned_data[video_id][
            'cleaned_right_vectors'
        ]
        cleaned_data[video_id].pop('cleaned_left_vectors')
        cleaned_data[video_id].pop('cleaned_right_vectors')
        cleaned_data[video_id].pop('left_vectors')
        cleaned_data[video_id].pop('right_vectors')

    ##############################################
    # asl_citizen_train_set creation
    ##############################################

    file_path = '/scratch/gpfs/mn4560/final/data/asl_citizen/splits/train.csv'
    video_to_gloss_dictionary_train = create_video_to_gloss_dict(file_path)

    ##############################################
    # asl_citizen_val_set creation
    ##############################################

    file_path = '/scratch/gpfs/mn4560/final/data/asl_citizen/splits/val.csv'
    video_to_gloss_dictionary_val = create_video_to_gloss_dict(file_path)

    ##############################################
    # asl_citizen_test_set creation
    ##############################################

    file_path = '/scratch/gpfs/mn4560/final/data/asl_citizen/splits/test.csv'
    video_to_gloss_dictionary_test = create_video_to_gloss_dict(file_path)

    #################################################
    # Add glosses to dictionary and convert to list of dictionaries
    # so we don't have issue with duplicates
    #################################################

    data_list = []
    for key in cleaned_data:
        left_positions_scaled = min_max_scale(cleaned_data[key]['leftpositions'])
        right_positions_scaled = min_max_scale(cleaned_data[key]['rightpositions'])

        left_displacements = calculate_displacements(left_positions_scaled)
        right_displacements = calculate_displacements(right_positions_scaled)

        left_displacements_centered = zero_mean_center(left_displacements)
        right_displacements_centered = zero_mean_center(right_displacements)

        temp_word = ''
        temp_dict = {}
        if str(key) in video_to_gloss_dictionary_train:
            temp_word = video_to_gloss_dictionary_train[str(key)]
        elif str(key) in video_to_gloss_dictionary_val:
            temp_word = video_to_gloss_dictionary_val[str(key)]
        elif str(key) in video_to_gloss_dictionary_test:
            temp_word = video_to_gloss_dictionary_test[str(key)]
        else:
            logging.warning(f'Invalid video ID {key}')
            continue

        temp_dict = {
            'word': temp_word,
            'positions': {
                'leftdisplacements': left_displacements_centered,
                'rightdisplacements': right_displacements_centered,
                'leftpositions': left_positions_scaled,
                'rightpositions': right_positions_scaled,
            },
        }
        data_list.append(temp_dict)

    # Convert data_list to DataFrame
    gloss_df = pd.DataFrame(data_list)

    # Shuffle the DataFrame
    gloss_df = gloss_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Define split sizes
    train_size = 0.5
    val_size = 0.1 / 0.5  # 10% of the whole dataset, but 20% of the 50% temp set
    test_size = 1 - val_size  # Remainder of the temp set

    # Splitting data into train and a temporary set (for validation and test)
    train_df, temp_df = train_test_split(gloss_df, test_size=0.5, random_state=42)

    # Splitting the temporary set into validation and test
    val_df, test_df = train_test_split(temp_df, test_size=test_size, random_state=42)

    # Save the datasets to new JSON files
    print('Creating train dataset...')
    with open(
        '/scratch/gpfs/mn4560/final/data/asl_citizen/asl_citizen_train_disp.json',
        'w',
    ) as f:
        json.dump(train_df.to_dict('records'), f, indent=4)

    print('Creating validation dataset...')
    with open(
        '/scratch/gpfs/mn4560/final/data/asl_citizen/asl_citizen_val_disp.json',
        'w',
    ) as f:
        json.dump(val_df.to_dict('records'), f, indent=4)

    print('Creating test dataset...')
    with open(
        '/scratch/gpfs/mn4560/final/data/asl_citizen/asl_citizen_test_disp.json',
        'w',
    ) as f:
        json.dump(test_df.to_dict('records'), f, indent=4)

    # Print the size of each set to verify correct proportions
    print(
        f'Train size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}'
    )
    print(
        f'Proportional split check -> Train: {len(train_df)/len(gloss_df):.2f}, Validation: {len(val_df)/len(gloss_df):.2f}, Test: {len(test_df)/len(gloss_df):.2f}'
    )
