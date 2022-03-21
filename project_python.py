"""
Search for certain text across many images of newspaper pages, and where the 
target text is found, return a contact sheet of images of any
faces found on that page.

Use pillow for image processing, tesseract for optical character recoginition,
and opencv for facial recognition.

University of Michigan
Python Project: pillow, tesseract, and opencv
Final Project
"""

import zipfile
from PIL import Image, ImageDraw
import pytesseract
import cv2
from kraken import pageseg
import numpy
from datetime import datetime
import string


# Warn user that the program takes a long time
msg = 'Warning: executing this program can take up to 25 minutes. '
msg += 'Are you sure you want to continue? (Y/N): '
decision = input(msg)
if decision.lower() == 'n':
    quit()

# Start Timer
start_main = datetime.now()
print(f"Execution Beginning (localbuild): ({start_main.time()})")

# loading the face detection classifier
face_cascade = cv2.CascadeClassifier(
    'readonly/haarcascade_frontalface_default.xml')


def extract_from_zip(zip_path, save_path):
    """
    Return a dictionary of the extracted files from the ZIP file path
    
    :param zip_path: string file path to a zip file
    :param save_path: string file path to directory of extracted files
    :return dicts: list of dictionaries for the extracted files, with:
        {name: file name,
        filepath: file path,
        img_pil: PIL.Image object of the image, in grayscale
    """

    print(f'\tRunning Function: extract_from_zip ({datetime.now().time()})')
    dicts = []
    with zipfile.ZipFile(zip_path, mode='r') as myzip:
        myzip.extractall(save_path)
        for file in myzip.infolist():
            d = {'name': file.filename, 'filepath': save_path + file.filename}
            d['img_pil'] = Image.open(d['filepath']).convert('L')
            dicts.append(d)
    return dicts


def get_text(d):
    """
    Use Pytesseract to read the text from the text boxes and the text to d
    """

    start = datetime.now()
    print(f"\t\tStarting Function: get_text ({d['name']}) ({start.time()})")
    text_boxes = []
    crop_box_pils = []
    for box in d['boxes']:
        crop_box_arr = d['img_proc_arr'][box[1]:box[3], box[0]:box[2]]
        crop_box_arr = process_image(crop_box_arr, img_type="arr", timeit=False)
        crop_box_pil = Image.fromarray(crop_box_arr).convert('1')
        crop_box_pils.append(crop_box_pil)
        new_text = pytesseract.image_to_string(crop_box_pil)
        new_text = new_text.lower()
        text_boxes.append(new_text)
    d['text_boxes'] = text_boxes
    d['crop_box_pils'] = crop_box_pils
    end = datetime.now()
    print(f'\t\tFunction Complete: get_text. Time elapsed: {end - start}')


def binarize(img_arr, thresh=177, timeit=True):
    """
    :param img: an ndarray image
    :param thresh: thresshold for binzarization (0 to 255)
    :return out_img: the binarized image as an ndarray
    """
    if timeit:
        startb = datetime.now()
        print(f"\t\tStarting Function: binarize ({startb.time()})")
    height, width = img_arr.shape
    out_arr = numpy.zeros((height, width))
    for x in range(width):
        for y in range(height):
            if img_arr[y][x] > thresh:
                out_arr[y][x] = 255
    
    if timeit:
        endb = datetime.now()
        print(f'\t\tFunction Complete: binarize. Time elapsed: {endb - startb}')
    return out_arr


def process_image(img, img_type='PIL', timeit=True):
    """
    Accepts an image, processes it, and returns a copy as a numpy array

    :param img: either a PIL image or a numpy array image
    :return img_proc_pil: the processed image as a numpy array
    """

    if img_type == 'PIL':
        imcv = numpy.array(img)
    elif img_type == 'arr':
        imcv = img.copy()
    else:
        print("ERROR :param img_type: for process_image must be PIL or arr.")
        print("process_image will exit without processing.")
        return
    imcv = binarize(imcv, thresh=177, timeit=timeit)
    imcv = cv2.resize(imcv, None, fx=1.75, fy=1.75, interpolation=cv2.INTER_CUBIC)
    kernel = numpy.ones((1, 1), numpy.uint8)
    imcv = cv2.dilate(imcv, kernel, iterations=1)
    imcv = cv2.erode(imcv, kernel, iterations=1)
    return imcv


# --- MAIN FUNCTION EXECUTION ---

# Extract the images from the ZIP File
test_args = ['readonly/small_img.zip', 'readonly/small_img/']
args = ['readonly/images.zip', 'readonly/images/']
#target_word = 'Christopher'
target_word = 'Mark'
page_dicts = extract_from_zip(*args)

for page_dict in page_dicts:
    start_d = datetime.now()
    print(f"\tCurrent Image: {page_dict['name']} ({start_d.time()})")

    # --- Process the Newspaper Image ---
    page_dict['img_proc_arr'] = process_image(page_dict['img_pil'])
    page_dict['img_proc_pil'] = Image.fromarray(page_dict['img_proc_arr']).convert('1')

    # --- Kraken --- Get Text Boxes ---
    startk = datetime.now()
    print(f"\t\tStarting Kraken ({startk.time()})")
    boxes = pageseg.segment(page_dict['img_proc_pil'])['boxes']
    page_dict['boxes'] = boxes
    endk = datetime.now()
    print(f'\t\tKraken Complete. Time elapsed: {endk - startk}')
    
    # --- Draw Boxes onto PIL Image ---
    page_dict['img_pil_boxes'] = page_dict['img_proc_pil'].copy()
    draw = ImageDraw.Draw(page_dict['img_pil_boxes'])
    for i, box in enumerate(page_dict['boxes']):
        draw.rectangle(box, outline='black')
    #page_dict['img_pil_boxes'].show()  --for debugging

    # --- Tesseract - Get Text Boxes ---
    get_text(page_dict)
    page_dict['text'] = "".join(page_dict['text_boxes']).replace(" ", "")

    # --- Search for Target Word ---
    page_dict['target_found'] = False
    if target_word.lower() in "".join(page_dict['text']).replace(" ", ""):
        page_dict['target_found'] = True

    end_d = datetime.now()
    print(f"\tFinished with Image {page_dict['name']}. Time elapsed: {end_d - start_d}")

end_main = datetime.now()
print(f'Execution Complete (localbuild). Time elapsed: {end_main - start_main}')

# Display results for a selected image(s) for review
for myd in page_dicts:
    print("\n-------------------------------------------\n")
    print(f"RESULTS FOR IMAGE: {myd['name']}")
    print("\tTarget Word Found?", myd['target_found'])
    print("\tNumber of boxes found:", len(myd['boxes']))
    print("\tLength of text:", len(myd['text']))
    print("\tContents of entire text: saved to text file in readonly")
    with open(f"readonly/small_img/{myd['name']}-txt-results.txt", 'w') as file:
        for i, text in enumerate(myd['text_boxes']):
            text.replace("\n", "")
            file.write(f'Box # {i}: {text}\n')
