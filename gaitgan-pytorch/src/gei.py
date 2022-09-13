from skimage.io import imread
from skimage.io import imsave
from scipy.misc import imresize
import numpy as np
import os
import logging
logger = logging.getLogger("tool")

def load_image_path_list(path):
    """
    :param path: the test image folder
    :return:
    """
    list_path = os.listdir(path)
    result = ["%s/%s" % (path, x) for x in list_path if x.endswith("jpg") or x.endswith("png")]
    return result

def image_path_list_to_image_pic_list(image_path_list):
    image_pic_list = []
    for image_path in image_path_list:
        im = imread(image_path)
        image_pic_list.append(im)
    return image_pic_list

def build_GEI(img_list):
    """
    :param img_list: a list of grey image numpy.array data
    :return:
    """
    norm_width = 64
    norm_height = 64
    result = np.zeros((norm_height, norm_width), dtype=np.int)

    human_extract_list = []
    for img in img_list:
        try:
            human_extract_list.append(img)
        except:
            logger.warning("fail to extract human from image")
    try:
        result = np.mean(human_extract_list, axis=0)
    except:
        logger.warning("fail to calculate GEI, return an empty image")

    return result.astype(np.int)

def img_path_to_GEI(img_path):
    """
    convert the images in the img_path to GEI
    :param img_path: string
    :return: a GEI image
    """
    id = img_path.replace("/", "_")
    img_list = load_image_path_list(img_path)
    img_data_list = image_path_list_to_image_pic_list(img_list)
    GEI_image = build_GEI(img_data_list)
    return GEI_image


if __name__ == '__main__':
    human_id = ["%03d" % i for i in range(75, 125)]#1, 125)]
    data_dir = ['bg-01', 'bg-02', 'cl-01', 'cl-02',
                     'nm-01', 'nm-02', 'nm-03', 'nm-04',
                     'nm-05', 'nm-06']
    view_list = ["%03d" % x for x in range(0, 181, 18)]
    casia_dataset_b_path = "../../GaitDatasetB_out"
    casia_dataset_b_gei_path = "../GaitDataseB_gei"
    for id in human_id:
        for dir in data_dir:
            for view in view_list:
                img_dir = "%s/%s/%s/%s" % (casia_dataset_b_path, id, dir, view)
                if not os.path.exists(img_dir):
                    logger.warning("%s do not exist" % img_dir)
                    continue
                if len(os.listdir(img_dir))!=0: 
                    GEI_image = img_path_to_GEI(img_dir)
                    #imsave("%s/GEI.bmp" % test_data_path, GEI_image)
                    save_dir = "%s/%s/" % (casia_dataset_b_gei_path, id)
                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)
                    save_dir = "%s/%s/%s/" % (casia_dataset_b_gei_path, id, dir)
                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)
                    save_img = os.path.join(save_dir,"%s-%s-%s.png" % (id, dir, view))
                    imsave(save_img, GEI_image)
    