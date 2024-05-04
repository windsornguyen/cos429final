import cv2
import os
import copy
import json
import csv
import multiprocessing
import logging
from tqdm import tqdm
import pickle

# Set up logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)
os.environ['GLOG_minloglevel'] = '2'
import mediapipe as mp  # Ensure you import mediapipe after setting the environment variable


# Global constants
CHECKPOINT_FILE = 'landmark_data_checkpoint.pkl'
MAX_WORKERS = 6
MISSING_VALUE = -1

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
                    current_frame_left, current_frame_right = [], []

                    if results.multi_hand_landmarks:
                        for hand_landmarks, handedness in zip(
                            results.multi_hand_landmarks,
                            results.multi_handedness,
                            strict=True,
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
        try:
            with open(checkpoint_file, 'rb') as f:
                features_dict = pickle.load(f)
                processed_files = set(features_dict.keys())
        except (EOFError, pickle.UnpicklingError):
            features_dict = {}
            processed_files = set()
    else:
        features_dict = {}
        processed_files = set()

    remaining_files = [
        filename for filename in video_files if filename not in processed_files
    ]

    print(f'Found {len(video_files)} video files in total.')
    print(f'Skipping {len(processed_files)} already processed videos.')
    print(f'Processing {len(remaining_files)} remaining videos...')

    with multiprocessing.Pool(processes=MAX_WORKERS) as pool:
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

    for result in results:
        if result is not None:
            features_dict[result['filename']] = result

    # Report checkpoint status
    num_processed = len([result for result in results if result is not None])
    total_processed = len(processed_files) + num_processed
    print(f'Processed {num_processed} videos in this run.')
    print(f'Total processed videos: {total_processed}')

    with open(checkpoint_file, 'wb') as f:
        pickle.dump(features_dict, f)

    return features_dict


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


def list_dataset(id_to_gloss_dictionary, full_dataset_dictionary):
    data_list = []
    for key in id_to_gloss_dictionary:
        if key in full_dataset_dictionary:
            temp_dict = {
                'word': id_to_gloss_dictionary[str(key)],
                'positions': {
                    'leftpositions': full_dataset_dictionary[key]['leftpositions'],
                    'rightpositions': full_dataset_dictionary[key]['rightpositions'],
                },
            }
            data_list.append(temp_dict)
    return data_list


def clean_landmark_data(landmark_data):
    ######################################
    # Observe that each video has extraneous frames where we have no
    # hands present at the start and end. We now clean the data by removing
    # these so each time-series only contains the data we actually care about
    ######################################

    for video_id in landmark_data:
        li = landmark_data[video_id]['numHands'].copy()
        first_non_zero_index = next((i for i, x in enumerate(li) if x != 0), 0)
        last_non_zero_index = next(
            (i for i, x in enumerate(reversed(li)) if x != 0), len(li) - 1
        )

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
    cleaned_data = copy.deepcopy(landmark_data)
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

    tuple_list = [(MISSING_VALUE, MISSING_VALUE) for _ in range(21)]

    for video_id in cleaned_data:
        cleaned_data[video_id]['leftpositions'] = [
            tuple_list if kp_vector == [] else kp_vector
            for kp_vector in cleaned_data[video_id]['leftpositions']
        ]
        cleaned_data[video_id]['rightpositions'] = [
            tuple_list if kp_vector == [] else kp_vector
            for kp_vector in cleaned_data[video_id]['rightpositions']
        ]

    return cleaned_data


def save_dataset(dataset, file_path):
    with open(file_path, 'w') as f:
        json.dump(dataset, f, indent=4)


def main():
    video_folder = '../ASL_Citizen/videos'
    landmark_data = process_videos_in_folder(video_folder, CHECKPOINT_FILE)
    cleaned_data = clean_landmark_data(landmark_data)

    train_file_path = '../ASL_Citizen/key_to_ids/ASL_Citizen_id_to_gloss_train.csv'
    val_file_path = '../ASL_Citizen/key_to_ids/ASL_Citizen_id_to_gloss_val.csv'
    test_file_path = '../ASL_Citizen/key_to_ids/ASL_Citizen_id_to_gloss_test.csv'

    video_to_gloss_dictionary_train = create_video_to_gloss_dict(train_file_path)
    video_to_gloss_dictionary_val = create_video_to_gloss_dict(val_file_path)
    video_to_gloss_dictionary_test = create_video_to_gloss_dict(test_file_path)

    ##############################################
    # asl_citizen_train_set creation
    ##############################################

    train_dataset = list_dataset(video_to_gloss_dictionary_train, cleaned_data)
    save_dataset(train_dataset, '../ASL_Citizen/asl_citizen_train.json')

    ##############################################
    # asl_citizen_val_set creation
    ##############################################

    val_dataset = list_dataset(video_to_gloss_dictionary_val, cleaned_data)
    save_dataset(val_dataset, '../ASL_Citizen/asl_citizen_val.json')

    ##############################################
    # asl_citizen_test_set creation
    ##############################################

    test_dataset = list_dataset(video_to_gloss_dictionary_test, cleaned_data)
    save_dataset(test_dataset, '../ASL_Citizen/asl_citizen_test.json')


if __name__ == '__main__':
    main()
