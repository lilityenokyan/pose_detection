import os
import requests
import PIL

from PIL import Image

SUCCESS_LIMIT = 100
FOLDER_PATH = 'dataset/yoga_poses/'


def download_file(url, path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)


def download_files_from_txt(txt_file):
    with open(txt_file, 'r') as file:
        lines = file.readlines()

    success_count = 0
    error_count = 0
    for line in lines:
        line = line.strip()
        if line:
            parts = line.split('\t')
            if len(parts) == 2:
                file_path, url = parts
                file_path = FOLDER_PATH + file_path
                folder, file_name = os.path.split(file_path)
                if not os.path.exists(folder):
                    os.makedirs(folder)
                file_name = os.path.join(folder, file_name)
                if not os.path.exists(file_name):
                    try:
                        download_file(url, file_name)
                        success_count += 1
                        print(f"Downloaded and verified: {file_name}")
                        if success_count >= SUCCESS_LIMIT:
                            print(f"Reached success limit of {SUCCESS_LIMIT} files. Moving on to the next TXT file.")
                            break
                    except (requests.exceptions.RequestException, OSError) as e:
                        error_count += 1
                        print(f"Error downloading {file_name}: {e}")
                    except Exception as e:
                        error_count += 1
                        print(f"Unknown error downloading {file_name}: {e}")
                    if error_count >= 10:
                        print("Reached error limit. Skipping remaining files.")
                        break
                else:
                    print(f"File already exists: {file_name}")
            else:
                print(f"Invalid line: {line}")


# Download the files in the corresponding folders using the urls from the .txt files
txt_files_folder = 'Yoga-82/yoga_dataset_links'
txt_files = [file for file in os.listdir(txt_files_folder) if file.endswith('.txt')]

for txt_file in txt_files:
    txt_file_path = os.path.join(txt_files_folder, txt_file)
    print(f"Processing TXT file: {txt_file}")
    download_files_from_txt(txt_file_path)


###################### These 2 functions can be used after download for filtration ######################
def is_valid_image(img_path):
    try:
        Image.open(img_path)
        return True
    except (FileNotFoundError, PIL.UnidentifiedImageError, OSError):
        return False


def delete_invalid_images(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            img_path = os.path.join(root, file)
            if not is_valid_image(img_path):
                print(f"Deleting invalid image: {img_path}")
                os.remove(img_path)


# Deletes the invalid images from the folder
# delete_invalid_images(FOLDER_PATH)
