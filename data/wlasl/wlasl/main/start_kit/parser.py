# Turns WLASL_v0.3.json into a CSV file with columns gloss and video_ids

import ujson as json
import csv

with open('data.json', 'r') as file:
    data = json.load(file)

csv_data = [['gloss', 'video_ids']]
for entry in data:
    gloss = entry['gloss']
    video_ids = [instance['video_id'] for instance in entry['instances']]
    csv_data.append([gloss] + video_ids)

with open('gloss_video_ids_new.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(csv_data)
