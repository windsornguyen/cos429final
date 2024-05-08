import os
import cv2
import mediapipe as mp
import msgspec
import logging
from multiprocessing import Pool, cpu_count

# Set up logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)


def process_video(input_file, output_dir):
    try:
        # Output file paths
        output_video_file = os.path.join(
            output_dir,
            os.path.splitext(os.path.basename(input_file))[0] + '_annotated.mp4',
        )
        output_data_file = os.path.join(
            output_dir, os.path.splitext(os.path.basename(input_file))[0] + '.msgpack'
        )

        if os.path.exists(output_data_file):
            logging.info(f'Skipping already processed file: {input_file}')
            return output_data_file

        # Initialize MediaPipe Hands model
        mp_hands = mp.solutions.hands
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
        ) as hands:
            # Open the input video
            cap = cv2.VideoCapture(input_file)
            if not cap.isOpened():
                logging.error(f'Failed to open video file: {input_file}')
                return None

            # Get video properties
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Create VideoWriter object to save the annotated video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                output_video_file, fourcc, fps, (frame_width, frame_height)
            )

            left_hand_vectors, right_hand_vectors = [], []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                current_frame_left, current_frame_right = [], []

                if results.multi_hand_landmarks:
                    for hand_landmarks, handedness in zip(
                        results.multi_hand_landmarks,
                        results.multi_handedness,
                        strict=True,
                    ):
                        # Draw landmarks on the frame
                        mp.solutions.drawing_utils.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                        )

                        hand_type = handedness.classification[0].label
                        landmarks = [
                            (landmark.x, landmark.y, landmark.z)
                            for landmark in hand_landmarks.landmark
                        ]

                        if hand_type == 'Left':
                            current_frame_left = landmarks
                        else:
                            current_frame_right = landmarks

                left_hand_vectors.append(current_frame_left)
                right_hand_vectors.append(current_frame_right)

                # Write the annotated frame to the output video
                out.write(frame)

            cap.release()
            out.release()

            # Save the data using msgspec.msgpack serialization
            with open(output_data_file, 'wb') as f:
                data = {
                    'left_positions': left_hand_vectors,
                    'right_positions': right_hand_vectors,
                }
                serialized_data = msgspec.msgpack.encode(data)
                f.write(serialized_data)

            logging.info(f'Successfully processed video: {input_file}')
            return output_data_file

    except Exception as e:
        logging.error(f'Error processing video {input_file}: {str(e)}')
        return None


def process_videos_in_directory(input_dir, output_dir, num_videos=200):
    try:
        os.makedirs(output_dir, exist_ok=True)

        video_files = [file for file in os.listdir(input_dir) if file.endswith('.mp4')][
            :num_videos
        ]

        # Create a multiprocessing pool
        pool = Pool(processes=cpu_count())

        # Process videos in parallel
        results = pool.starmap(
            process_video,
            [
                (os.path.join(input_dir, input_file), output_dir)
                for input_file in video_files
            ],
        )

        # Close the pool
        pool.close()
        pool.join()

        processed_videos = len([result for result in results if result is not None])
        logging.info(f'Processed {processed_videos} videos out of {num_videos}')
        logging.info(f'Finished processing videos in directory: {input_dir}')

    except Exception as e:
        logging.error(f'Error processing videos in directory {input_dir}: {str(e)}')
