import json

with open('../dataset/Yoga-82/yoga_test.txt', 'r') as f:
    lines = f.readlines()

pose_dict = {}
for line in lines:
    parts = line.strip().split(',')
    img_path = parts[0].split('/')[0].replace('_', ' ')  # Remove underscores from the text
    label = int(parts[-1])
    pose_dict[label] = img_path

# Save the dictionary to a JSON
with open('../dataset/Yoga-82/yoga_labels82.json', 'w') as json_file:
    json.dump(pose_dict, json_file)
