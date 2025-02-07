import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import cv2
import os
import logging
import sys
from skimage.feature import SIFT
from skimage import exposure
from skimage.color import rgb2gray, gray2rgb
from sklearn.manifold import TSNE


DEBUG = False


def configure_logging(debug_mode=False):
    log_level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout
    )

logger = logging.getLogger(__name__)
configure_logging(DEBUG)




def image_preprocessing(img_path = './output/rgb', output_path: str = './output/sift', 
                        rotation_aug = False, experiment_with_gray_scale = True, use_entire_dataset = True, 
                        load_all_images = False, use_height_map = True, distraction_merge = False, 
                        distraction_merge_to_one = False, original_merge_to_one = False, 
                        rotation_aug_lower = 150, rotation_aug_upper = 350,
                        nonrotation_aug_lower = 50, nonrotation_aug_upper = 450,
                        molecular_imprinting_name = 'MP'):

    data_dir = pathlib.Path(img_path)
    image_count = len(list(data_dir.glob('*.jpg')))
    logging.debug(f"image count = {image_count}")


    def sift_features_vector(src, image_path, random_keypoints_upper: int = 1000, height_map=None):
        src = rgb2gray(src)
        img_adapteq = exposure.equalize_adapthist(src, clip_limit=0.03)
        print(img_adapteq.shape)
        descriptor_extractor = SIFT()
        # descriptor_extractor = ORB(n_keypoints=50)
        descriptor_extractor.detect_and_extract(img_adapteq)
        keypoints = descriptor_extractor.keypoints
        descriptors = descriptor_extractor.descriptors
        
        # random select 100 keypoints
        random_keypoints = np.random.randint(0, len(keypoints), random_keypoints_upper) # 200 for DMMP, 500 for MP, 1000 for MP
        keypoints = keypoints[random_keypoints]
        descriptors = descriptors[random_keypoints]
        
        height_value = []
        if height_map is not None:
            for i in range(len(keypoints)):
                height = height_map[int(keypoints[i][0]), int(keypoints[i][1])]
                height_value.append(height)
        
        plt.imshow(img_adapteq, cmap='gray')
        plt.scatter(keypoints[:, 1], keypoints[:, 0],
                    facecolors='none', edgecolors='r')
        plt.savefig('./output/sift/%s' % (image_path))
        plt.close()
        if height_map is not None:
            descriptors = np.hstack((descriptors, np.array(height_value).reshape(-1, 1)))
        print(descriptors.shape)
        return descriptors

    # %%
    def generate_height_map(size, offset: int = 315):
        size = size + 2*offset

        r = 1.0

        # Create a 2D grid of x and y coordinates
        x, y = np.meshgrid(np.linspace(-r, r, size), np.linspace(-r, r, size))

        # Calculate the corresponding z coordinates
        # Note: For points outside the sphere, this will be NaN
        z = np.sqrt(r**2 - x**2 - y**2)

        # We set points outside the sphere to zero height for visualization
        z[np.isnan(z)] = 0

        z = z[offset:size-offset, offset:size-offset]

        return z
    

    # # visualise the height map in 3D
    # height_map = generate_height_map(400)

    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection='3d')

    # # Create the x, y, and z coordinate arrays. We use 
    # # numpy's broadcasting to do all the hard work for us.
    # # We could shorten this even more by using np.meshgrid.
    # x = np.arange(height_map.shape[0])
    # y = np.arange(height_map.shape[1])
    # x, y = np.meshgrid(x, y)
    # z = height_map

    # # Load the image
    # image = cv2.imread('./output/rgb/M-DMMP- NaBF4-10-6M-1.jpg')
    # image = rgb2gray(image)
    # image = exposure.equalize_adapthist(image, clip_limit=0.03)
    # image = gray2rgb(image)
    # height, width = image.shape[:2]
    # resize_image = cv2.resize(src=image, dsize=(int(width / 2), int(height / 2)))
    # resize_image = resize_image[50:450, 50:450]

    # # Map the image to the surface
    # ax.plot_surface(x, y, z, facecolors=image, shade=False)

    # plt.show()



    os.makedirs(output_path, exist_ok=True)
    x = []
    y = []
    for path in data_dir.glob('*.jpg'):
        if not load_all_images:
            if path.name.split('-')[1] != molecular_imprinting_name and path.name.split('-')[1].split('(')[0] != molecular_imprinting_name:
                    continue
        src = cv2.imread(str(path))
        height, width = src.shape[:2]
        center = (width / 2, height / 2)
        if rotation_aug:
            for i in range(3):
                rotation_matrix = rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=90 * i, scale=1)
                rotated_image = cv2.warpAffine(src=src, M=rotate_matrix, dsize=(width, height))
                resize_image = cv2.resize(src=src, dsize=(int(width / 2), int(height / 2)))
                # crop image 200x200
                resize_image = resize_image[rotation_aug_lower:rotation_aug_upper, rotation_aug_lower:rotation_aug_upper]
                if use_height_map:
                    height_map = generate_height_map(200)
                    resize_image = np.dstack((resize_image.astype(np.float32), height_map))
                feature_vector = sift_features_vector(resize_image, os.path.basename(path))
                x.append(feature_vector)
                y.append(path.name.split('-')[2].replace(' ', ''))
        else:
            resize_image = cv2.resize(src=src, dsize=(int(width / 2), int(height / 2)))
            # crop image 200x200
            resize_image = resize_image[nonrotation_aug_lower:nonrotation_aug_upper, nonrotation_aug_lower:nonrotation_aug_upper]
            if use_height_map:
                height_map = generate_height_map(400)
            else:
                height_map = None
            feature_vector = sift_features_vector(resize_image, os.path.basename(path), height_map=height_map)
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
    x = x.reshape(x.shape[0], -1)
    logging.debug(x.shape)
    logging.debug(y.shape)
    return x, y


class data_mining: 
    def method_1(self, x, y, use_entire_dataset = True):
        # TODO check we need to keep these codes
        # from sklearn.preprocessing import StandardScaler
        # sc = StandardScaler()
        # sc.fit(x)
        #
        # x = sc.transform(x)
        # x_test = sc.transform(x_test)
        # x_train = sc.transform(x_train)
        pca = TSNE(n_components=3, learning_rate='auto', init='pca', perplexity=3, random_state=1)
        if use_entire_dataset:
            x_train_pca = pca.fit_transform(x)
            y_train = y
        # else:
        #     x_train_pca = pca.fit_transform(x_train)
        #     y_train = y_train
        logging.debug(x_train_pca)

        
        # TODO check we need to keep these codes
        # print('total explain ratio=%s' % sum(pca.explained_variance_ratio_[:3]))
        # input('%s %s total explain ratio=%s' % (
        #     molecular_imprinting_name, str((int(1000 / pixels_per_cell[0]), int(1000 / pixels_per_cell[0]))), sum(pca.explained_variance_ratio_[:3])))


        n_components=3
        df = pd.DataFrame(x_train_pca, columns=[f"PC{i + 1}" for i in range(n_components)])
        label_list = []

        for i in range(len(x_train_pca)):
            label_list.append(y_train[i])
            # if y[i] == 'DMMP':
            #     label_list.append('MP, DMMP, MPA')
            # elif y[i] == 'NaBF4':
            #     label_list.append('NaBF4')
            # elif y[i] == 'MP':
            #     label_list.append('MP, DMMP, MPA')
            # elif y[i] == 'MPA':
            #     label_list.append('MP, DMMP, MPA')
            # elif y[i] == 'Original':
            #     label_list.append('Original')
            # elif y[i] == 'KF6P':
            #     label_list.append('KF6P')
            # elif y[i] == 'MP, DMMP, MPA':
            #     label_list.append('MP, DMMP, MPA')
            # else:
            #     print('ERROR')
        df['label'] = label_list
        logging.debug(df)
        return df


class data_visualization:
    def __init__(self):
        logging.debug(pio.templates)
        pio.templates.default = 'plotly'


    def method_1(self, df, molecular_imprinting_name):
        fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3', color='label', symbol='label', title='%s SIFT' % (molecular_imprinting_name), color_discrete_map={
                        "DMMP": "red",
                        "NaBF4": "green",
                        "KF6P": "blue",
                        "MPA": "goldenrod",
                        "MP": "purple"},
                        symbol_sequence= ['circle', 'circle', 'circle', 'circle'])
        fig.update_traces(marker=dict(size=6),selector=dict(mode='markers'))
        fig.update_layout(scene=dict(
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(showticklabels=False)
        ))
        fig.show()


if __name__ == "__main__":
    molecular_imprinting_name = 'MP'
    data_miner = data_mining()
    data_visual = data_visualization()

    x, y = image_preprocessing()
    df = data_miner.method_1(x, y)
    data_visual.method_1(df, molecular_imprinting_name)

