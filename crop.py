import cv2
import os
import sys
import shutil
from tqdm import tqdm

def crop_faces_directory(directory_path, tolerance=0, num_images=None):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    filenames = os.listdir(directory_path)
    if num_images is not None:
        filenames = filenames[:num_images]

    output_folder_name = f"{os.path.basename(directory_path)}_Cropped"
    output_folder = os.path.join(directory_path, output_folder_name)
    os.makedirs(output_folder, exist_ok=True)

    count = 0
    for filename in tqdm(filenames, desc='Processing images'):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            file_path = os.path.join(directory_path, filename)
            img = cv2.imread(file_path)

            faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

            max_face_size = 0
            max_face_coords = None

            for (x, y, w, h) in faces:
                face_size = w * h

                if face_size > max_face_size:
                    max_face_size = face_size
                    max_face_coords = (x, y, w, h)

            if max_face_coords is not None:
                (x, y, w, h) = max_face_coords

                margin = int((w + h) * 0.5 * tolerance)
                x -= margin
                y -= margin
                w += 2 * margin
                h += 2 * margin

                face = img[y:y+h, x:x+w]

                cropped_path = os.path.join(output_folder, 'cropped_' + filename)
                cv2.imwrite(cropped_path, face)
                count += 1

            if num_images is not None and count == num_images:
                break

    os.chdir(directory_path)
    print(f"All images in '{directory_path}' have been processed and cropped. {count} images have been saved in '{output_folder_name}' directory.")

def crop_face_file(file_path, tolerance=0):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    img = cv2.imread(file_path)

    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print('No faces were found.')
        return

    largest_face_idx = 0
    largest_face_area = 0
    for i, (x, y, w, h) in enumerate(faces):
        if w * h > largest_face_area:
            largest_face_area = w * h
            largest_face_idx = i

    x, y, w, h = faces[largest_face_idx]

    margin = int((w + h) * 0.5 * tolerance)
    x -= margin
    y -= margin
    w += 2 * margin
    h += 2 * margin

    if x < 0:
        w += x
        x = 0
    if y < 0:
        h += y
        y = 0
    if x + w > img.shape[1]:
        w = img.shape[1] - x
    if y + h > img.shape[0]:
        h = img.shape[0] - y

    face = img[y:y+h, x:x+w]

    cropped_path = os.path.join(os.path.dirname(file_path), 'cropped_' + os.path.basename(file_path))
    cv2.imwrite(cropped_path, face)

    cropped_img = cv2.imread(cropped_path)
    cv2.imshow('Cropped Face', cropped_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
