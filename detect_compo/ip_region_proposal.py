import cv2
from os.path import join as pjoin
import time
import json
import numpy as np

import detect_compo.lib_ip.ip_preprocessing as pre
import detect_compo.lib_ip.ip_draw as draw
import detect_compo.lib_ip.ip_detection as det
import detect_compo.lib_ip.file_utils as file
import detect_compo.lib_ip.Component as Compo
from config.CONFIG_UIED import Config
C = Config()


def nesting_inspection(org, grey, compos, grad_min, ffl_block):
    '''
    Inspect all big compos through block division by flood-fill
    :param ffl_block: gradient threshold for flood-fill
    :return: nesting compos
    '''
    nesting_compos = []
    for i, compo in enumerate(compos):
        if compo.height > 10:
        #if compo.height > 10 and i==6:
            replace = False
            clip_org = compo.compo_clipping(org)
            #cv2.imwrite('/home/auto-test-4/wyx/datasets/UIED_dataset/output/grey.jpg', org)
            #cv2.imwrite('/home/auto-test-4/wyx/datasets/UIED_dataset/output/clip_org.jpg', clip_org)
            #import pdb
            #pdb.set_trace()
            clip_org_binary = pre.nest_binarization(clip_org, grad_min)
            #print(clip_org_binary)
            #cv2.imwrite('/home/auto-test-4/wyx/datasets/UIED_dataset/output/clip_org_binary.jpg', clip_org_binary)
            n_compos = det.nested_components_detection(clip_org_binary, org, grad_thresh=ffl_block, show=False)
            #print(n_compos)
            Compo.cvt_compos_relative_pos(n_compos, compo.bbox.col_min, compo.bbox.row_min)

            for n_compo in n_compos:
                if n_compo.redundant:
                    compos[i] = n_compo
                    replace = True
                    break
            if not replace:
                nesting_compos += n_compos
    return nesting_compos


def compo_detection(input_img_path, output_root, uied_params,
                    resize_by_height=800, classifier=None, show=False, wai_key=0):

    start = time.clock()
    name = input_img_path.split('/')[-1][:-4] if '/' in input_img_path else input_img_path.split('\\')[-1][:-4]
    ip_root = file.build_directory(pjoin(output_root, "ip"))
    clf_root = file.build_directory(pjoin(output_root, "clf"))

    # *** Step 1 *** pre-processing: read img -> get binary map
    org, grey = pre.read_img(input_img_path, resize_by_height)
    binary = pre.binarization(org, grad_min=int(uied_params['min-grad']))
    #cv2.save('have-line', binary)
    cv2.imwrite('data/output/binary_with_line.jpg', binary)

    # *** Step 2 *** element detection
    det.rm_line(binary, show=show, wait_key=wai_key)
    cv2.imwrite('data/output/binary_no_line.jpg', binary)

    uicompos = det.component_detection(binary, min_obj_area=int(uied_params['min-ele-area']))
    #import pdb
    #pdb.set_trace()

    # *** Step 3 *** results refinement
    #print(len(uicompos))
    uicompos = det.compo_filter(uicompos, min_area=int(uied_params['min-ele-area']), img_shape=binary.shape)
    #print(len(uicompos))
    uicompos = det.merge_intersected_compos(uicompos)
    #print(len(uicompos))
    det.compo_block_recognition(binary, uicompos)
    if uied_params['merge-contained-ele']:
        uicompos = det.rm_contained_compos_not_in_block(uicompos)
    Compo.compos_update(uicompos, org.shape)
    Compo.compos_containment(uicompos)

    # *** Step 4 ** nesting inspection: check if big compos have nesting element
    #uicompos += nesting_inspection(org, grey, uicompos, grad_min=int(uied_params['min-grad']), ffl_block=uied_params['ffl-block'])
    
    #print(len(uicompos))
    uicompos += nesting_inspection(org, grey, uicompos, grad_min=int(uied_params['min-grad']), ffl_block=uied_params['ffl-block'])
    #print(len(uicompos))
    uicompos += nesting_inspection(org, grey, uicompos, grad_min=int(uied_params['min-grad']), ffl_block=uied_params['ffl-block'])
    #print(len(uicompos))

    '''
    uicompos_nesting1 = nesting_inspection(org, grey, uicompos, grad_min=int(uied_params['min-grad']), ffl_block=uied_params['ffl-block'])
    uicompos += uicompos_nesting1
    print(len(uicompos_nesting1))
    uicompos_nesting2 = nesting_inspection(org, grey, uicompos_nesting1, grad_min=int(uied_params['min-grad']), ffl_block=uied_params['ffl-block'])#二次嵌套检查
    print(len(uicompos_nesting2))
    uicompos_nesting3 = nesting_inspection(org, grey, uicompos, grad_min=int(uied_params['min-grad']), ffl_block=uied_params['ffl-block'])#二次嵌套检查
    print(len(uicompos_nesting3))
    uicompos += uicompos_nesting3
    '''
    
    Compo.compos_update(uicompos, org.shape)
    draw.draw_bounding_box(org, uicompos, show=show, name='merged compo', write_path=pjoin(ip_root, name + '.jpg'), wait_key=wai_key)

    #*** Step 5 *** image inspection: recognize image -> remove noise in image -> binarize with larger threshold and reverse -> rectangular compo detection
    # if classifier is not None:
    #     classifier['Image'].predict(seg.clipping(org, uicompos), uicompos)
    #     draw.draw_bounding_box_class(org, uicompos, show=show)
    #     uicompos = det.rm_noise_in_large_img(uicompos, org)
    #     draw.draw_bounding_box_class(org, uicompos, show=show)
    #     det.detect_compos_in_img(uicompos, binary_org, org)
    #     draw.draw_bounding_box(org, uicompos, show=show)
    # if classifier is not None:
    #     classifier['Noise'].predict(seg.clipping(org, uicompos), uicompos)
    #     draw.draw_bounding_box_class(org, uicompos, show=show)
    #     uicompos = det.rm_noise_compos(uicompos)

    #*** Step 6 *** element classification: all category classification
    if classifier is not None:
        classifier['Elements'].predict([compo.compo_clipping(org) for compo in uicompos], uicompos)
        draw.draw_bounding_box_class(org, uicompos, show=show, name='cls', write_path=pjoin(clf_root, name + '.jpg'))
        #draw.draw_bounding_box_class(org, uicompos, write_path=pjoin(clf_root, 'result.jpg'))

    # *** Step 7 *** save detection result
    Compo.compos_update(uicompos, org.shape)
    file.save_corners_json(pjoin(ip_root, name + '.json'), uicompos)
    print("[Compo Detection Completed in %.3f s] Input: %s Output: %s" % (time.clock() - start, input_img_path, pjoin(ip_root, name + '.json')))
