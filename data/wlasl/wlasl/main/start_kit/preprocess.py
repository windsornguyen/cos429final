import os
import msgspec
import cv2
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def convert_to_mp4(file_path):
    name, ext = os.path.splitext(file_path)
    output_path = f'{name}.mp4'

    if ext == '.swf':
        cmd = f'ffmpeg -i "{file_path}" -c:v libx264 -crf 23 -preset medium -c:a aac -b:a 128k -movflags +faststart -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" "{output_path}" -loglevel error -y'
    elif ext == '.mkv':
        cmd = f'ffmpeg -i "{file_path}" -c:v libx264 -crf 23 -preset medium -c:a aac -b:a 128k -movflags +faststart "{output_path}" -loglevel error -y'
    else:
        return

    subprocess.call(cmd, shell=True)

    if os.path.exists(output_path):
        os.remove(file_path)

def convert_everything_to_mp4():
    cmd = 'bash scripts/swf2mp4.sh'
    os.system(cmd)

def video_to_frames(video_path, size=None):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if size:
            frame = cv2.resize(frame, size)

        frames.append(frame)

    cap.release()
    return frames

def convert_frames_to_video(frame_array, path_out, size, fps=25):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path_out, fourcc, fps, size)

    for frame in frame_array:
        out.write(frame)

    out.release()

def extract_frame_as_video(src_video_path, start_frame, end_frame):
    frames = video_to_frames(src_video_path)
    return frames[start_frame: end_frame+1]

def process_video(inst, src_video_path, dst_video_path, cnt):
    url = inst['url']
    video_id = inst['video_id']

    if 'youtube' in url or 'youtu.be' in url:
        cnt += 1
        
        if not os.path.exists(src_video_path):
            return cnt

        if os.path.exists(dst_video_path):
            print('{} exists.'.format(dst_video_path))
            return cnt

        # because the JSON file indexes from 1.
        start_frame = inst['frame_start'] - 1
        end_frame = inst['frame_end'] - 1

        if end_frame <= 0:
            shutil.copyfile(src_video_path, dst_video_path)
        else:
            selected_frames = extract_frame_as_video(src_video_path, start_frame, end_frame)
            
            # when OpenCV reads an image, it returns size in (h, w, c)
            # when OpenCV creates a writer, it requires size in (w, h).
            size = selected_frames[0].shape[:2][::-1]
            
            convert_frames_to_video(selected_frames, dst_video_path, size)

        print(cnt, dst_video_path)
    else:
        cnt += 1

        if os.path.exists(dst_video_path):
            print('{} exists.'.format(dst_video_path))
            return cnt

        if not os.path.exists(src_video_path):
            return cnt

        print(cnt, dst_video_path)
        shutil.copyfile(src_video_path, dst_video_path)

    return cnt

def extract_all_yt_instances(content):
    cnt = 1

    if not os.path.exists('videos'):
        os.mkdir('videos')

    with ProcessPoolExecutor() as executor:
        futures = []
        for entry in content:
            instances = entry['instances']

            for inst in instances:
                video_id = inst['video_id']
                
                if 'youtube' in inst['url'] or 'youtu.be' in inst['url']:
                    yt_identifier = inst['url'][-11:]
                    src_video_path = os.path.join('raw_videos_mp4', yt_identifier + '.mp4')
                else:
                    src_video_path = os.path.join('raw_videos_mp4', video_id + '.mp4')
                
                dst_video_path = os.path.join('videos', video_id + '.mp4')

                future = executor.submit(process_video, inst, src_video_path, dst_video_path, cnt)
                futures.append(future)

        for future in tqdm(futures, desc='Extracting YouTube instances'):
            cnt = future.result()

def main():
    # 1. Convert .swf, .mkv file to mp4.
    convert_everything_to_mp4()

    # Load JSON data using msgspec
    with open('WLASL_v0.3.json', 'rb') as f:
        content = msgspec.json.decode(f.read())

    # 2. Extract YouTube frames and create video instances.
    extract_all_yt_instances(content)

if __name__ == '__main__':
    main()
