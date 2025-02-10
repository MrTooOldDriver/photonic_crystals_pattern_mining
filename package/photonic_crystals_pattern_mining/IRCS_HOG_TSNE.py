import pathlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.io as pio
import plotly.express as px
import logging
import sys
from skimage.feature import hog
from sklearn.manifold import TSNE
from skimage import exposure
from sklearn.model_selection import train_test_split




def image_preprocessing(img_path: str = './output/rgb', output_path: str = './output/sift', 
                        rotation_aug = False, experiment_with_gray_scale = True, 
                        use_entire_dataset = True, enhance = True, crop = False, 
                        load_all_images = False, distraction_merge = False, 
                        distraction_merge_to_one = False, original_merge_to_one = False, 
                        pixels_per_cell = (100, 100), orientations = 9, cells_per_block = (1, 1), 
                        ksize = (5,5), crop_size = 350, molecular_imprinting_name = 'DMMP', 
                        random_keypoints_upper: int = 2500, offset: int = 315):
    # get all files ending with .jpg
    data_dir = pathlib.Path(img_path)
    image_count = len(list(data_dir.glob('*.jpg')))
    logging.debug(f"image count = {image_count}")

    def hog_feature_vector(src, pixels_per_cell, orientations, cells_per_block):
        fd = hog(src, orientations = orientations, pixels_per_cell = pixels_per_cell, cells_per_block = cells_per_block, channel_axis=2)
        # fd = hog(src, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), channel_axis=2)
        return fd

    def image_enhance(src, ksize):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
        # Top Hat Transform
        topHat = cv2.morphologyEx(src, cv2.MORPH_TOPHAT, kernel)
        # Black Hat Transform
        blackHat = cv2.morphologyEx(src, cv2.MORPH_BLACKHAT, kernel)
        src = src + topHat - blackHat
        # print('enhance on')
        return src



    x = []
    y = []
    for path in data_dir.glob('*.jpg'):
        if not load_all_images:
            if path.name.split('-')[1] != molecular_imprinting_name and path.name.split('-')[1].split('(')[0] != molecular_imprinting_name:
                continue
        src = cv2.imread(str(path))
        if experiment_with_gray_scale:
            src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            src = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)
        else:
            src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        height, width = src.shape[:2]
        center = (width / 2, height / 2)

        if crop:
            height, width = src.shape[:2]
            center = [int(width / 2), int(height / 2)]
            resize_image = src[center[0]-crop_size:center[0]+crop_size, center[1]-crop_size:center[1]+crop_size]

        if rotation_aug:
            for i in range(3):
                rotation_matrix = rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=90 * i, scale=1)
                rotated_image = cv2.warpAffine(src=src, M=rotate_matrix, dsize=(width, height))
                resize_image = cv2.resize(src=src, dsize=(int(width / 2), int(height / 2)))
                if enhance:
                    resize_image = image_enhance(resize_image, ksize)
                feature_vector = hog_feature_vector(resize_image, pixels_per_cell, orientations, cells_per_block)
                x.append(feature_vector)
                y.append(path.name.split('-')[2].replace(' ', ''))
        else:
            if enhance:
                src = image_enhance(src, ksize)
            feature_vector = hog_feature_vector(src, pixels_per_cell, orientations, cells_per_block)
            x.append(feature_vector)
            if load_all_images:
                label_name = path.name.split('-')[1].split('(')[0] + '-' + path.name.split('-')[2].replace(' ', '')
                if label_name.split('-')[0] != label_name.split('-')[1]:
                    if distraction_merge or distraction_merge_to_one:
                        if distraction_merge_to_one:
                            label_name = 'distraction'
                        else:
                            label_name = label_name.split('-')[0] + '-distraction'
                elif label_name.split('-')[0] == label_name.split('-')[1]:
                    if original_merge_to_one :
                        label_name = 'original'
            else:
                if distraction_merge or distraction_merge_to_one:
                    if path.name.split('-')[2].split('(')[0] != molecular_imprinting_name:
                        if distraction_merge_to_one:
                            label_name = 'distraction'
                        else:
                            label_name = path.name.split('-')[1].split('(')[0] + '-distraction'
                    else:
                        label_name = path.name.split('-')[2].replace(' ', '')
                else:
                    label_name = path.name.split('-')[2].replace(' ', '')
            y.append(label_name)

    x = np.array(x)
    y = np.array(y)
    logging.debug('data loaded x=%i' % (len(x)))
    logging.debug('data loaded y=%i' % (len(y)))

    return x, y


class data_mining:
    def method_1(self, experiment_with_gray_scale = True, enhance = True, crop_size = 350, pixels_per_cell = (100, 100), image_dir = 'output/rgb/M-MPA-KF6P-10-6M-1.jpg'):
        img = cv2.imread(image_dir)
        if experiment_with_gray_scale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if enhance:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            # Top Hat Transform
            topHat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
            # Black Hat Transform
            blackHat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
            img = img + topHat - blackHat

        height, width = img.shape[:2]
        center = [int(width / 2), int(height / 2)]
        resize_image = img[center[0]-crop_size:center[0]+crop_size, center[1]-crop_size:center[1]+crop_size]

        fd, hog_image = hog(resize_image, orientations=9, pixels_per_cell=pixels_per_cell, cells_per_block=(1, 1), visualize=True, channel_axis=2)
        return resize_image, hog_image
    

    def method_2(self, x, y, x_train, y_train, use_entire_dataset = True, perplexity = 3):
        pca = TSNE(n_components=3, learning_rate='auto', init='pca', perplexity=perplexity)
        if use_entire_dataset:
            x_train_pca = pca.fit_transform(x)
            y_train = y
        else:
            x_train_pca = pca.fit_transform(x_train)
            y_train = y_train
        logging.debug(x_train_pca)

        n_components=3
        df = pd.DataFrame(x_train_pca, columns=[f"PC{i + 1}" for i in range(n_components)])
        label_list = []

        for i in range(len(x_train_pca)):
            label_list.append(y_train[i])
        df['label'] = label_list
        logging.debug(df)
        return df


class data_visualization:
    def __init__(self):
        print(pio.templates)
        pio.templates.default = 'plotly'


    def method_1(self, x, y, resize_image, hog_image):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15))

        ax1.axis('off')
        ax1.imshow(resize_image)
        ax1.set_title('Input image')

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 1))

        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        plt.show()


    def method_2(self, df, molecular_imprinting_name, perplexity = 3, pixels_per_cell = (100, 100)):
        fig = px.scatter_3d(df, x='PC1', y='PC2',z='PC3',  color='label', symbol='label', title='Hog TSNE %s Granularity=%s Sperplexity=%s' % (molecular_imprinting_name, str((int(1000 / pixels_per_cell[0]), int(1000 / pixels_per_cell[0]))), str(perplexity)), color_discrete_map={
                        "DMMP": "red",
                        "NaBF4": "green",
                        "KF6P": "blue",
                        "MPA": "goldenrod",
                        "MP": "purple"},
                        symbol_sequence= ['circle', 'circle', 'circle', 'circle'])
        fig.update_layout(scene=dict(
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(showticklabels=False)
        ))
        fig.show()







if __name__ == "__main__":
    DEBUG = False


    def configure_logging(debug_mode=False):
        log_level = logging.DEBUG if debug_mode else logging.INFO
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            stream=sys.stdout
        )

    logger = logging.getLogger(__name__)
    debug_mode = True
    configure_logging(debug_mode)
    molecular_imprinting_name = 'DMMP'
    data_miner = data_mining()
    data_visual = data_visualization()

    x, y = image_preprocessing()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, stratify=y)
    logging.debug(x_train.shape)
    logging.debug(y_train)
    logging.debug(y_test)
    resize_image, hog_image = data_miner.method_1()
    data_visual.method_1(x, y, resize_image, hog_image)
    df = data_miner.method_2(x, y, x_train, y_train)
    data_visual.method_2(df, molecular_imprinting_name)


# TODO check if this code is useful

# # img = cv2.imread('output/rgb/M-DMMP(190nm)- NaBF4-10-2M-1.jpg')
# img = cv2.imread('./output/rgb/M-DMMP- NaBF4-10-6M-1.jpg')
# # img = cv2.imread('output/rgb/M-MP(180nm)-DMMP-10-2M-1.jpg')
# fd = hog(img, orientations=9, pixels_per_cell=(30, 30), cells_per_block=(1, 1), channel_axis=2)
# print(fd.shape)







