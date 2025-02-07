import pathlib
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import logging
import sys
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler





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

def image_preprocessing(img_path: str = './output/no_bg_fix_circle_mask_canny', rotation_aug = False, 
                        load_all_images = False, resize_factor = 50, distraction_merge = False, 
                        distraction_merge_to_one = False, original_merge_to_one = False, 
                        molecular_imprinting_name = 'DMMP'):
    
    data_dir = pathlib.Path(img_path)
    image_count = len(list(data_dir.glob('*.jpg')))
    logging.debug(image_count)

    x = []
    y = []
    resize_x = []
    for path in data_dir.glob('*.jpg'):
        if not load_all_images:
            if path.name.split('-')[1] != molecular_imprinting_name and path.name.split('-')[1].split('(')[0] != molecular_imprinting_name:
                continue
        src = cv2.imread(str(path))
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        height, width = src.shape[:2]
        center = (width / 2, height / 2)
        if rotation_aug:
            for i in range(3):
                rotation_matrix = rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=90 * i, scale=1)
                rotated_image = cv2.warpAffine(src=src, M=rotate_matrix, dsize=(width, height))
                resize_image = cv2.resize(src=rotated_image, dsize=(int(width / resize_factor), int(height / resize_factor)))
                resize_x.append(resize_image)
                resize_image = np.reshape(resize_image, (resize_image.shape[0] * resize_image.shape[1]))
                x.append(resize_image)
                if load_all_images:
                    label_name = path.name.split('-')[1].split('(')[0] + '-' + path.name.split('-')[2].replace(' ', '')
                    y.append(label_name)
                else:
                    y.append(path.name.split('-')[2].replace(' ', ''))
        else:
            resize_image = cv2.resize(src=src, dsize=(int(width / resize_factor), int(height / resize_factor)))
            resize_x.append(resize_image)
            resize_image = np.reshape(resize_image, (resize_image.shape[0] * resize_image.shape[1]))
            x.append(resize_image)
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
    resize_x = np.array(resize_x)
    logging.debug('data loaded x=%i' % (len(x)))
    logging.debug('data loaded y=%i' % (len(y)))
    return x, y, resize_x



class data_mining:
    def method_1(self, x, y):
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        use_entire_dataset = True

        perplexity=5
        pca = TSNE(n_components=3, learning_rate='auto', init='pca', perplexity=perplexity)
        if use_entire_dataset:
            x_train_pca = pca.fit_transform(x)
            y_train = y
        # else:
        #     x_train_pca = pca.fit_transform(x_train)
        #     y_train = y_train

        logging.debug(x_train_pca)


        df = pd.DataFrame(x_train_pca, columns=[f"PC{i + 1}" for i in range(3)])
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


    def method_1(self, df, molecular_imprinting_name, resize_x, perplexity = 5):
        fig = px.scatter_3d(df, x='PC1', y='PC2',z='PC3',  color='label', symbol='label', title='Canny TSNE %s Granularity=%s Sperplexity=%s' % (molecular_imprinting_name, str(resize_x[0].shape), str(perplexity)), color_discrete_map={
                        "DMMP": "red",
                        "NaBF4": "green",
                        "KF6P": "blue",
                        "MPA": "goldenrod",
                        "MP": "purple"},
                        symbol_sequence= ['circle', 'circle', 'circle', 'circle'],)
        fig.update_layout(scene=dict(
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(showticklabels=False)
        ))
        fig.show()





# TODO check if this code is useful
# white_mask = np.ones(shape=resize_x[0].shape)
# print('Granularity=%s' % str(resize_x[0].shape))
# plt.figure(figsize=(10, 10))
# plt.imshow(white_mask - resize_x[0], cmap='gray')



if __name__ == "__main__":
    molecular_imprinting_name = 'DMMP'
    data_miner = data_mining()
    data_visual = data_visualization()

    x, y, resize_x = image_preprocessing()
    df = data_miner.method_1(x, y)
    data_visual.method_1(df, molecular_imprinting_name, resize_x)