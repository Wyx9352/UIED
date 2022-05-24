from os.path import join as pjoin
import cv2
import os
import numpy as np
from cnn.CNN import CNN
from cnn.Data import Data


#def resize_height_by_longest_edge(img_path, resize_length=800):
    org = cv2.imread(img_path)
    height, width = org.shape[:2]
    if height > width:
        return resize_length
    else:
        return int(resize_length * (height / width))

'''
def color_tips():
    color_map = {'Text': (0, 0, 255), 'Compo': (0, 255, 0), 'Block': (0, 255, 255), 'Text Content': (255, 0, 255)}
    board = np.zeros((200, 200, 3), dtype=np.uint8)

    board[:50, :, :] = (0, 0, 255)
    board[50:100, :, :] = (0, 255, 0)
    board[100:150, :, :] = (255, 0, 255)
    board[150:200, :, :] = (0, 255, 255)
    cv2.putText(board, 'Text', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(board, 'Non-text Compo', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(board, "Compo's Text Content", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(board, "Block", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.imshow('colors', board)
'''


if __name__ == '__main__':
    

    model = CNN(classifier_type = 'Elements', is_load=False)#调用keras的ResNet50模型

    # 给出训练和测试数据
    data = Data()
    print('X_train shape : ', data.X_train.shape)
    print('Y_train shape : ', data.Y_train.shape)
    print('X_test shape : ', data.X_test.shape)
    print('Y_test shape : ', data.Y_test.shape)

    model.train(data)

    # 画图看一下训练的效果
    plt.plot(training.history['acc'])
    plt.plot(training.history['loss'])
    plt.title('model accuracy and loss')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['acc', 'loss'], loc='upper left')
    plt.savefig('Model/wyx/rico/train.jpg')
    #plt.show()

    model.evaluate(data)

    '''
    key_params = {'min-grad':10, 'ffl-block':5, 'min-ele-area':50,
                  'merge-contained-ele':True, 'merge-line-to-paragraph':False, 'remove-bar':True}

    # set input image path
    input_path_img = 'data/input/IMG_0091.PNG'
    output_root = 'data/output'

    resized_height = resize_height_by_longest_edge(input_path_img, resize_length=800)
    color_tips()

    is_ip = True
    is_clf = True
    #is_ocr = False
    #is_merge = False
    
    if is_ocr:
        import detect_text.text_detection as text
        os.makedirs(pjoin(output_root, 'ocr'), exist_ok=True)
        text.text_detection(input_path_img, output_root, show=True, method='google')
    '''

    '''
     if classifier is not None:
        classifier['Elements'].predict([compo.compo_clipping(org) for compo in uicompos], uicompos)
        draw.draw_bounding_box_class(org, uicompos, show=show, name='cls', write_path=pjoin(clf_root, name + '.jpg'))
        #draw.draw_bounding_box_class(org, uicompos, write_path=pjoin(clf_root, 'result.jpg'))
    
    if is_merge:
        import detect_merge.merge as merge
        os.makedirs(pjoin(output_root, 'merge'), exist_ok=True)
        name = input_path_img.split('/')[-1][:-4]
        compo_path = pjoin(output_root, 'ip', str(name) + '.json')
        ocr_path = pjoin(output_root, 'ocr', str(name) + '.json')
        merge.merge(input_path_img, compo_path, ocr_path, pjoin(output_root, 'merge'),
                    is_remove_bar=key_params['remove-bar'], is_paragraph=key_params['merge-line-to-paragraph'], show=True)
    '''
