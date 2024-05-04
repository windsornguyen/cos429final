#################
# IMPORTS
#################

import cv2
import mediapipe as mp
import os
import copy
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import multiprocessing
import logging
from tqdm import tqdm
import pickle

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
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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
                    current_frame_left, current_frame_right = [], []

                    if results.multi_hand_landmarks:
                        for hand_landmarks, handedness in zip(
                            results.multi_hand_landmarks,
                            results.multi_handedness,
                            strict=False,
                        ):
                            hand_type = handedness.classification[0].label
                            landmarks = [
                                (landmark.x, landmark.y)
                                for landmark in hand_landmarks.landmark
                            ]

                            if hand_type == 'Left':
                                current_frame_left.extend(landmarks)
                            else:
                                current_frame_right.extend(landmarks)

                    left_hand_vectors.append(current_frame_left)
                    right_hand_vectors.append(current_frame_right)

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


if __name__ == '__main__':
    # Specify the folder containing the videos
    video_folder = '../wlasl/videos'  # Modify this path as necessary
    checkpoint_file = 'landmark_data_checkpoint.pkl'
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

    #############################################
    # Currently there are a lot of empty coordinates (for when only one hand is
    # detectable at that point). This is annoying for model to deal with so replace
    # every empty list with list of length 21 that has (-1,-1).
    #############################################

    C = -1
    tuple_list = [(C, C) for _ in range(21)]

    for video_id in cleaned_data:
        cleaned_data[video_id]['leftpositions'] = [
            tuple_list if kp_vector == [] else kp_vector
            for kp_vector in cleaned_data[video_id]['leftpositions']
        ]
        cleaned_data[video_id]['rightpositions'] = [
            tuple_list if kp_vector == [] else kp_vector
            for kp_vector in cleaned_data[video_id]['rightpositions']
        ]

    #########################################
    # CSV key_id to gloss script for WLASL database
    #########################################

    def create_id_to_gloss_dict(file_path):
        id_to_gloss = {}

        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.read().splitlines()
            for line in lines[1:]:  # Skip the header line
                parts = line.strip().split(',')
                gloss = parts[0]
                video_ids = parts[1:]
                for video_id in video_ids:
                    id_to_gloss[str(video_id) + '.mp4'] = gloss

        return id_to_gloss

    # Specify the path to the CSV file
    file_path = '../wlasl/WLASL_id_to_gloss.csv'

    # Create the dictionary
    id_to_gloss_dictionary = create_id_to_gloss_dict(file_path)

    #################################################
    # Add glosses to dictionary and convert to list of dictionaries
    # so we don't have issue with duplicates
    #################################################

    data_list = []
    for key in cleaned_data:
        temp_dict = {
            'word': id_to_gloss_dictionary[str(key)],
            'positions': {
                'leftpositions': cleaned_data[key]['leftpositions'],
                'rightpositions': cleaned_data[key]['rightpositions'],
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
    with open('../wlasl/wlasl_train.json', 'w') as f:
        json.dump(train_df.to_dict('records'), f, indent=4)

    print('Creating validation dataset...')
    with open('../wlasl/wlasl_val.json', 'w') as f:
        json.dump(val_df.to_dict('records'), f, indent=4)

    print('Creating test dataset...')
    with open('../wlasl/wlasl_test.json', 'w') as f:
        json.dump(test_df.to_dict('records'), f, indent=4)

    # Print the size of each set to verify correct proportions
    print(
        f'Train size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}'
    )
    print(
        f'Proportional split check -> Train: {len(train_df)/len(gloss_df):.2f}, Validation: {len(val_df)/len(gloss_df):.2f}, Test: {len(test_df)/len(gloss_df):.2f}'
    )