from As.src.Detetction.detection import Detection
from As.src.Recognition.recognition import Recognition
import numpy as np
from glob import glob
import cv2


def System(input_images):
    """"
    Our System which takes images that need to be converted to text
    """
    # An array which contain the cropped images of each input image
    # At index i there will be a list of cropped images for the ith input image
    # {'image_name': [[[]]]}
    input_cropped_images, imgs_line_begin_indices = Detection(input_images)


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
    print( '/n------------------------------------------------------this is System------------------------------------------------------' )
    print(text_outputs)
    print( '/n------------------------------------------------------this is System------------------------------------------------------')

    return text_outputs,#, by_line_inputs_text


if __name__ == '__main__':
    # paths = glob('src/Recognition/test_images/*')
    # print(len(paths))
    paths = glob('new_tests/*')
    input_imgs = []
    for path in paths:
        input_imgs.append(cv2.imread(path))
    # o1 = Recognition([input_imgs])
    o1 = System(input_imgs)
    print(o1)
    print('yes')