import multiprocessing
import glob
import time
import json
import os
from tqdm import tqdm
from os.path import join as pjoin, exists
import cv2

import detect_compo.ip_region_proposal as ip


def resize_height_by_longest_edge(img_path, resize_length=800):
    org = cv2.imread(img_path)
    height, width = org.shape[:2]
    if height > width:
        return resize_length
    else:
        return int(resize_length * (height / width))


if __name__ == '__main__':
    # initialization
    input_img_root = "/home/auto-test-4/wyx/datasets/UIED_dataset/input/screenshots"
    output_root = "/home/auto-test-4/wyx/datasets/UIED_dataset/output/screenshots"
    #data = json.load(open('E:/Mulong/Datasets/rico/instances_test.json', 'r'))

    #input_imgs = [pjoin(input_img_root, img['file_name'].split('/')[-1]) for img in data['images']]
    #input_imgs = sorted(input_imgs, key=lambda x: int(x.split('/')[-1][:-4]))  # sorted by index
    input_imgs = []
    #for filename in os.listdir(input_img_root):              #listdir的参数是文件夹的路径
    #    input_imgs.append[pjoin(input_img_root,filename)]
    input_imgs = [pjoin(input_img_root, img) for img in os.listdir(input_img_root)]
    #import pdb
    #pdb.set_trace()
    print(input_imgs)


    key_params = {'min-grad': 10, 'ffl-block': 5, 'min-ele-area': 50, 'merge-contained-ele': True,
                  'max-word-inline-gap': 10, 'max-line-ingraph-gap': 4, 'remove-top-bar': True}

    is_ip = True
    is_clf = True
    is_ocr = False
    is_merge = False

    # Load deep learning models in advance
    compo_classifier = None
    if is_ip and is_clf:
        compo_classifier = {}
        from cnn.CNN import CNN
        # compo_classifier['Image'] = CNN('Image')
        compo_classifier['Elements'] = CNN('Elements')
        # compo_classifier['Noise'] = CNN('Noise')
    ocr_model = None
    if is_ocr:
        import detect_text.text_detection as text

    # set the range of target inputs' indices
    num = 0
    start_index = 30800  # 61728
    end_index = 100000
    for input_img in input_imgs:
        resized_height = resize_height_by_longest_edge(input_img)
        '''
        index = input_img.split('/')[-1][:-4]
        if int(index) < start_index:
            continue
        if int(index) > end_index:
            break
        '''
        if is_ocr:
            text.text_detection(input_img, output_root, show=False)

        if is_ip:
            ip.compo_detection(input_img, output_root, key_params,  classifier=compo_classifier, resize_by_height=resized_height, show=False)

        if is_merge:
            import detect_merge.merge as merge
            compo_path = pjoin(output_root, 'ip', str(index) + '.json')
            ocr_path = pjoin(output_root, 'ocr', str(index) + '.json')
            merge.merge(input_img, compo_path, ocr_path, output_root, is_remove_top=key_params['remove-top-bar'], show=True)

        num += 1
