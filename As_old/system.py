from As_old.src.Detetction.detection import Detection
from As_old.src.Recognition.recognition import Recognition
import numpy as np
from glob import glob
import cv2
import torch
import matplotlib.pyplot as plt

def System(input_images):
    print(len(input_images))

    """"
    Our System which takes images that need to be converted to text
    """
    # An array which contain the cropped images of each input image
    # At index i there will be a list of cropped images for the ith input image
    # {'image_name': [[[]]]}
    input_cropped_images, imgs_line_begin_indices = Detection(input_images)
    print(input_cropped_images)

    # Here we send image to the recognition model to get the text
    inputs_text = Recognition(input_cropped_images)

    # Here make each line in an input image in a single list
    # the output of each input image is list of lines containing list of words
    by_line_inputs_text = []
    text_outputs = []
    for i, input_text in enumerate(inputs_text):
        text_output = ''
        for line in np.split(np.array(input_text), imgs_line_begin_indices[i]):
            text_output += " ".join(list(line))
            text_output += '\n'
        text_outputs.append(text_output)
        input_lines = [list(line) for line in np.split(np.array(input_text), imgs_line_begin_indices[i])]
        by_line_inputs_text.append(input_lines)

    # {'image_name': "text inside the image"}
    # finally here we return all the text to the website
    return text_outputs #, by_line_inputs_text


if __name__ == '__main__':
    paths = glob('test_images/*.tiff')
    input_imgs = []
    for path in paths:
        input_imgs.append(cv2.imread(path))
        break
    o1 = System(input_imgs)
    print(o1)
    print('yes')