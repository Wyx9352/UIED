## How to use?

### Dependency
* **Python 3.5**
* **Opencv 3.4.2**
* **Pandas**
<!-- * **Tensorflow 1.10.0**
* **Keras 2.2.4**
* **Sklearn 0.22.2** -->

### Installation
<!-- Install the mentioned dependencies, and download two pre-trained models from [this link](https://drive.google.com/drive/folders/1MK0Om7Lx0wRXGDfNcyj21B0FL1T461v5?usp=sharing) for EAST text detection and GUI element classification. -->

<!-- Change ``CNN_PATH`` and ``EAST_PATH`` in *config/CONFIG.py* to your locations. -->

The new version of UIED equipped with Google OCR is easy to deploy and no pre-trained model is needed. Simply donwload the repo along with the dependencies.

> Please replace the Google OCR key at `detect_text/ocr.py line 28` with your own (apply in [Google website](https://cloud.google.com/vision)).
### Usage
To test your own image(s):
* To test single image, change *input_path_img* in ``run_single.py`` to your input image and the results will be output to *output_root*.
* To test mutiple images, change *input_img_root* in ``run_batch.py`` to your input directory and the results will be output to *output_root*.
* To adjust the parameters lively, using ``run_testing.py`` 

> Note: The best set of parameters vary for different types of GUI image (Mobile App, Web, PC). I highly recommend to first play with the ``run_testing.py`` to pick a good set of parameters for your data.
   
## Folder structure
``cnn/``
* Used to train classifier for graphic UI elements
* Set path of the CNN classification model

``config/``
* Set data paths 
* Set parameters for graphic elements detection

``data/``
* Input UI images and output detection results

``detect_compo/``
* Non-text GUI component detection

``detect_text/``
* GUI text detection using Google OCR

``detect_merge/``
* Merge the detection results of non-text and text GUI elements

The major detection algorithms are in ``detect_compo/``, ``detect_text/`` and ``detect_merge/``