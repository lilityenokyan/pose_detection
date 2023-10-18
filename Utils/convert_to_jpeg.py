import os
from PIL import Image


def convert_jpg_to_jpeg_in_folder(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".png"):
                jpg_image_path = os.path.join(root, file)
                relative_path = os.path.relpath(jpg_image_path, input_folder)
                jpeg_image_path = os.path.join(output_folder, relative_path.lower().replace(".jpg", ".jpeg"))
                # Create the output folder
                output_subfolder = os.path.dirname(jpeg_image_path)
                os.makedirs(output_subfolder, exist_ok=True)

                # Convert the image to JPEG and save it
                with Image.open(jpg_image_path) as img:
                    img = img.convert("RGB")
                    img.save(jpeg_image_path, format="JPEG")


if __name__ == "__main__":
    input_folder = "../dataset/yoga_poses/"
    output_folder = "../dataset/yoga_poses_jpeg/"
    convert_jpg_to_jpeg_in_folder(input_folder, output_folder)
