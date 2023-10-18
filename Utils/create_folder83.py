import os
import random
from PIL import Image


def crop_random_parts(images_folder, output_folder, num_crops=500, crop_width=100, crop_height=100):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [filename for filename in os.listdir(images_folder) if filename.endswith('.jpg')]

    for i in range(num_crops):
        random_image_file = random.choice(image_files)
        image_path = os.path.join(images_folder, random_image_file)
        image = Image.open(image_path)
        width, height = image.size

        if width < crop_width or height < crop_height:
            continue

        random_x = random.randint(0, width - crop_width)
        random_y = random.randint(0, height - crop_height)

        cropped_image = image.crop((random_x, random_y, random_x + crop_width, random_y + crop_height))
        output_filename = f"cropped_{i}.png"
        output_path = os.path.join(output_folder, output_filename)

        cropped_image.save(output_path)


if __name__ == "__main__":
    # Randomly chosen folder to generate the cropped images from
    images_folder = "../dataset/yoga_poses/Cobra_Pose_or_Bhujangasana_"
    output_folder = "../dataset/yoga_poses/not_yoga"

    crop_random_parts(images_folder, output_folder, num_crops=500, crop_width=100, crop_height=100)
