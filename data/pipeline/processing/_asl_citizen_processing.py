#################
# IMPORTS
#################

import cv2
import mediapipe as mp
import os
import copy
import csv

#####################
# Extract features from ASL videos via keypoints for LHS and RHS.
# We remove noise and inaccuracies in the keypoint models by only storing data
# from every other frame. This gives us a time series for each video where we know the 
# (1) number of hands in the frame, (2) keypoints for the left hand, (3) 
# keypoints for the right hand
######################

def process_video(input_file):
    # Initialize the Mediapipe Hands solution
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=2,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Open the video file
    cap = cv2.VideoCapture(input_file)

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
            num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
            num_hands_list.append(num_hands)
            current_frame_left, current_frame_right = [], []

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness, strict=False):
                    hand_type = handedness.classification[0].label
                    landmarks = [(landmark.x, landmark.y) for landmark in hand_landmarks.landmark]

                    if hand_type == 'Left':
                        current_frame_left.extend(landmarks)
                    else:
                        current_frame_right.extend(landmarks)

            left_hand_vectors.append(current_frame_left)
            right_hand_vectors.append(current_frame_right)

        frame_count += 1

    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()

    # Return the landmark data
    return {
        'numHands': num_hands_list,
        'left_vectors': left_hand_vectors,
        'right_vectors': right_hand_vectors
    }

def process_videos_in_folder(folder_path):
    features_dict = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.mp4'):  # Check for video files
            input_video = os.path.join(folder_path, filename)
            print(f'Processing {input_video}...')
            landmarks_data = process_video(input_video)
            features_dict[filename] = landmarks_data
            print(f'Data collection complete for {input_video}.')

    return features_dict

# Specify the folder containing the videos
video_folder = '../ASL_Citizen/videos'  # Modify this path as necessary
landmark_data = process_videos_in_folder(video_folder)
# print("Landmark data collected for each video:", landmark_data)

######################################
# Observe that each video has extraneous frames where we have no 
# hands present at the start and end. We now clean the data by removing 
# these so each time-series only contains the data we actually care about
######################################


# Assuming landmark_data is defined
for video_id in landmark_data:
  li = landmark_data[video_id]['numHands'].copy()
  first_non_zero_index = 0
  last_non_zero_index = len(li) - 1

  # Finding first non-zero entry
  for i in range(len(li)):
    if li[i] == 0:
      first_non_zero_index += 1
    else:
      break

  # Finding last non-zero entry
  for i in range(len(li) - 1, 0, -1):
    if li[i] == 0:
      last_non_zero_index -= 1
    else:
      break

  # Cleaning rhs and lhs vectors based on first and last non_zero entry
  start = first_non_zero_index
  end = last_non_zero_index
  landmark_data[video_id]['start'] = start
  landmark_data[video_id]['end'] = end
  landmark_data[video_id]['cleaned_left_vectors'] = landmark_data[video_id]['left_vectors'][start : end + 1]
  landmark_data[video_id]['cleaned_right_vectors'] = landmark_data[video_id]['right_vectors'][start : end + 1]

# Copy everything so we have correct data
cleaned_data = copy.deepcopy(landmark_data)  # Use deepcopy to ensure full independence
for video_id in cleaned_data:
  cleaned_data[video_id].pop('numHands')
  cleaned_data[video_id].pop('start')
  cleaned_data[video_id].pop('end')
  cleaned_data[video_id]['leftpositions'] = cleaned_data[video_id]['cleaned_left_vectors']
  cleaned_data[video_id]['rightpositions'] = cleaned_data[video_id]['cleaned_right_vectors']
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
    for index, kp_vector in enumerate(cleaned_data[video_id]['leftpositions']):
        if kp_vector == []:
            cleaned_data[video_id]['leftpositions'][index] = tuple_list
    for index, kp_vector in enumerate(cleaned_data[video_id]['rightpositions']):
        if kp_vector == []:
            cleaned_data[video_id]['rightpositions'][index] = tuple_list

# print(cleaned_data)

##############################################
# general set creation helpers
##############################################


# cleaning hi1 and hi2 to both be hi
def remove_trailing_numbers(s):
    # While the last character is a digit, remove it
    while s and s[-1].isdigit():
        s = s[:-1]
    return s

def create_video_to_gloss_dict(file_path):
    # Dictionary to hold the mapping of video files to gloss terms
    video_to_gloss = {}

    # Open the CSV file and create a dictionary reader
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
            # print(key)
            # print(type(key))
            temp_dict = {
                'word' : id_to_gloss_dictionary[str(key)],
                'positions' : {
                    'leftpositions' : full_dataset_dictionary[key]['leftpositions'],
                    'rightpositions' : full_dataset_dictionary[key]['rightpositions']
                }
            }
            data_list.append(temp_dict)
    return data_list

##############################################
# asl_citizen_train_set creation
##############################################


file_path = '../ASL_Citizen/key_to_ids/ASL_Citizen_id_to_gloss_train.csv'
video_to_gloss_dictionary_train = create_video_to_gloss_dict(file_path)
train_dataset = list_dataset(video_to_gloss_dictionary_train, cleaned_data)

with jsonlines.open('../ASL_Citizen/asl_citizen_train.jsonl', 'w') as writer:
    writer.write_all(train_dataset)

# print("RAN A")
# print(train_dataset)
# print("TRAIN ON")

##############################################
# asl_citizen_val_set creation
##############################################

file_path = '../ASL_Citizen/key_to_ids/ASL_Citizen_id_to_gloss_val.csv'
video_to_gloss_dictionary_val = create_video_to_gloss_dict(file_path)
val_dataset = list_dataset(video_to_gloss_dictionary_val, cleaned_data)

with jsonlines.open('../ASL_Citizen/asl_citizen_val.jsonl', 'w') as writer:
    writer.write_all(val_dataset)

##############################################
# asl_citizen_test_set creation
##############################################

file_path = '../ASL_Citizen/key_to_ids/ASL_Citizen_id_to_gloss_test.csv'
video_to_gloss_dictionary_test = create_video_to_gloss_dict(file_path)
test_dataset = list_dataset(video_to_gloss_dictionary_test, cleaned_data)

with jsonlines.open('../ASL_Citizen/asl_citizen_test.jsonl', 'w') as writer:
    writer.write_all(test_dataset)
