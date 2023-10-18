import os
import cv2
import json
import torch
import numpy as np
import math
import mediapipe as mp
import torchvision.transforms as transforms
import torch.nn.functional as F

# Preprocess the image
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize the MediaPipe solution objects
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_image = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

cutoff_threshold = 10  # head and face
new_connections = list([t for t in mp_pose.POSE_CONNECTIONS if t[0] > cutoff_threshold and t[1] > cutoff_threshold])


# Filter the list of joints to avoid facial landmarks, rearrange the rest given new connections
def joints_connected_face_filtered():
    pairs = []

    for i in range(len(new_connections)):
        for j in range(i + 1, len(new_connections)):
            if set(new_connections[i]) & set(new_connections[j]):
                p1, p2 = new_connections[i], new_connections[j]
                common_value = None

                # Find the common value
                for value in p1:
                    if value in p2:
                        common_value = value
                        break

                # Rearrange the values with common_value in the middle, the 1st and 3rd in increasing order
                values = sorted(list(set(p1) | set(p2)))
                values.remove(common_value)
                values.insert(1, common_value)

                rearranged_pair = tuple(values)
                pairs.append(rearranged_pair)

    return pairs


def calculate_inner_angle(point1, point2, point3):
    det = point2.y * (point1.x - point3.x) + point1.y * (point3.x - point2.x) + point3.y * (point2.x - point1.x)
    det = det + point2.z * (point1.y - point3.y) + point1.z * (point3.y - point2.y) + point3.z * (point2.y - point1.y)

    a = np.array([point1.x, point1.y, point1.z])
    b = np.array([point2.x, point2.y, point2.z])
    c = np.array([point3.x, point3.y, point3.z])
    ba = a - b
    bc = c - b
    dot = np.dot(ba, bc)

    if dot == 0:
        return math.pi / 2
    else:
        angle = math.atan(det / dot)
    return angle


def calculate_similarity(frame_landmarks, ground_truth_landmarks):
    # Get the list of the joints with a common connection
    connected_joints = joints_connected_face_filtered()
    frame_angles = []
    ground_truth_angles = []
    # Iterate over connected_joints in the frame pose and the ground truth pose
    for pair in connected_joints:
        frame_a = frame_landmarks.landmark[pair[0]]
        frame_b = frame_landmarks.landmark[pair[1]]
        frame_c = frame_landmarks.landmark[pair[2]]

        ground_truth_b = ground_truth_landmarks.landmark[pair[1]]
        ground_truth_a = ground_truth_landmarks.landmark[pair[0]]
        ground_truth_c = ground_truth_landmarks.landmark[pair[2]]

        frame_angle = calculate_inner_angle(frame_a, frame_b, frame_c)
        frame_angles.append(frame_angle)

        ground_truth_angle = calculate_inner_angle(ground_truth_a, ground_truth_b, ground_truth_c)
        ground_truth_angles.append(ground_truth_angle)

    frame_angles = np.array(frame_angles).reshape(1, -1)
    ground_truth_angles = np.array(ground_truth_angles).reshape(1, -1)

    # Calculate the circular correlation coefficient
    cosine_similarities = np.cos(frame_angles - ground_truth_angles)
    correlation_coefficient = np.mean(cosine_similarities)
    return correlation_coefficient * 100


def get_image_path(label):
    label_str = str(label)
    label_with_underscores = label_str.replace(' ', '_')
    image_folder = os.path.join('dataset/yoga_poses_selected/', label_with_underscores)
    image_name = os.listdir(image_folder)[0]
    image_path = os.path.join(image_folder, image_name)
    return image_path


# Resize the image to fit the lower right corner of the frame
def resize_predicted_image(predicted_image, frame_size):
    height, width, _ = frame_size
    # Calculate the new size while maintaining the aspect ratio
    predicted_image_height, predicted_image_width, _ = predicted_image.shape
    max_new_height = int(height / 3)
    max_new_width = int(width / 3)
    # Calculate the scaling factor
    scale_factor = min(max_new_height / predicted_image_height, max_new_width / predicted_image_width)
    new_height = int(predicted_image_height * scale_factor)
    new_width = int(predicted_image_width * scale_factor)
    x_offset = width - new_width - 10
    y_offset = height - new_height - 10
    # Add the landmarks
    predicted_image = cv2.resize(predicted_image, (new_width, new_height))
    image_in_rgb = cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB)
    predicted_pose_landmarks = pose_image.process(image_in_rgb).pose_landmarks

    mp_drawing.draw_landmarks(
        image=predicted_image,
        landmark_list=predicted_pose_landmarks,
        connections=mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(5, 200, 5), thickness=2, circle_radius=2)
    )
    return predicted_image, x_offset, y_offset, predicted_pose_landmarks


def classification_and_detection(model):
    # Load the dictionary from the JSON file
    with open('dataset/Yoga-82/yoga_labels82.json', 'r') as json_file:
        label_dict = json.load(json_file)
    # Access camera
    # To access the iphone camera from mac, try 0, for default mac camera try 1
    cap = cv2.VideoCapture(0)  # default camera
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    while True:
        ret, frame = cap.read()

        # Get body landmarks
        image_in_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pose_landmarks = pose_image.process(image_in_rgb).pose_landmarks
        low_visibility = False
        similarity = None

        if frame_pose_landmarks is None:
            label = "Low visibility"
            low_visibility = True
        else:
            # Check for the visibility of landmarks, if it is less than 0.7 do NOT try classification
            # Ignore facial landmarks (idz>10)
            landmarks = frame_pose_landmarks.landmark
            low_visibility = [(idx, landmark.visibility) for idx, landmark in enumerate(landmarks, 1) if
                              (landmark.visibility < 0.2) and (idx > 10.0)]
            label = "Low visibility" if low_visibility else ""

        if not low_visibility:
            # Preprocess the frame for classification
            input_tensor = transform(frame)
            input_batch = input_tensor.unsqueeze(0)

            # Make a prediction using the pre-trained model
            with torch.no_grad():
                output = model(input_batch)

            probabilities = F.softmax(output, dim=1)
            predicted_class_prob, predicted_class_idx = torch.max(probabilities, 1)

            # If the predictions do not have high probability or predicted class is 82(non-yoga poses)
            # give "unknown pose" label
            if predicted_class_prob.item() < 0.8 or predicted_class_idx.item() == 82:
                label = "Unknown Pose"
            else:
                # Load the image corresponding to the predicted label
                predicted_class_label = label_dict[str(predicted_class_idx.item())]
                image_path = get_image_path(predicted_class_label)
                predicted_image = cv2.imread(image_path)
                resized_predicted_image, x_offset, y_offset, predicted_pose_landmarks = resize_predicted_image(
                    predicted_image, frame.shape)

                # Add the predicted image to the frame at the calculated position
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=frame_pose_landmarks,
                    connections=mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=3, circle_radius=3),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(50, 50, 237), thickness=2, circle_radius=2)
                )

                similarity = calculate_similarity(frame_pose_landmarks, predicted_pose_landmarks)
                frame[y_offset:y_offset + resized_predicted_image.shape[0],
                x_offset:x_offset + resized_predicted_image.shape[1]] = resized_predicted_image
                # label = f"Predicted Class: {predicted_class_label} | Probability: {predicted_class_prob:.4f} | Similarity: {similarity:.4f}"
                label = f"Predicted Pose: {predicted_class_label}"
                similarity_label = f"Similarity: {similarity:.4f}"

        font_scale = 2
        font_thickness = 8
        text_color = (0, 0, 0)  # (B, G, R)
        font = cv2.FONT_HERSHEY_DUPLEX

        similarity_position = (10, 140)
        label_position = (10, 70)

        cv2.putText(frame, label, label_position, font, font_scale, text_color, font_thickness, cv2.LINE_4)
        if similarity:
            cv2.putText(frame, similarity_label, similarity_position, font, font_scale, (0, 0, 200), font_thickness,
                        cv2.LINE_4)
        cv2.imshow('Live Classification', frame)

        # Exit and release the camera when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture
    cap.release()
    cv2.destroyAllWindows()


def detection_with_image_upload(ground_truth_path):
    # Access camera
    # To access the iphone camera from mac, try 0, for default mac camera try 1
    cap = cv2.VideoCapture(0)  # default camera
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    while True:
        ret, frame = cap.read()

        # Get body landmarks
        image_in_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pose_landmarks = pose_image.process(image_in_rgb).pose_landmarks

        ground_truth = cv2.imread(ground_truth_path)
        resized_ground_truth_image, x_offset, y_offset, predicted_pose_landmarks = resize_predicted_image(ground_truth, frame.shape)
        frame[y_offset:y_offset + resized_ground_truth_image.shape[0],
        x_offset:x_offset + resized_ground_truth_image.shape[1]] = resized_ground_truth_image

        if frame_pose_landmarks is None:
            label = "Low visibility"
            low_visibility = True
        else:
            # Check for the visibility of landmarks, if it is less than 0.7 do NOT try classification
            # Ignore facial landmarks (idz>10)
            landmarks = frame_pose_landmarks.landmark
            low_visibility = [(idx, landmark.visibility) for idx, landmark in enumerate(landmarks, 1) if
                              (landmark.visibility < 0.5) and (idx > 10.0)]
            label = "Low visibility" if low_visibility else ""

        if not low_visibility:

            # Add the true image to the frame at the calculated position
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=frame_pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=3, circle_radius=3),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(50, 50, 237), thickness=2, circle_radius=2)
            )

            similarity = calculate_similarity(frame_pose_landmarks, predicted_pose_landmarks)
            label = f"Similarity: {similarity:.4f}"

        font_scale = 3
        font_thickness = 8
        text_color = (0, 0, 0)  # (B, G, R)
        font = cv2.FONT_HERSHEY_DUPLEX

        cv2.putText(frame, label, (10, 70), font, font_scale, text_color, font_thickness, cv2.LINE_4)
        cv2.imshow('Live Classification', frame)

        # Exit and release the camera when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture
    cap.release()
    cv2.destroyAllWindows()
