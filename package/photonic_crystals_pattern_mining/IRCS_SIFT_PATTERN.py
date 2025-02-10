import pathlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.graph_objects as go
import os
import logging
import sys
from skimage.feature import SIFT
from skimage import exposure
from skimage.color import rgb2gray, gray2rgb






def image_preprocessing(img_path = './output/rgb', output_path: str = './output/sift', 
                        rotation_aug = False, experiment_with_gray_scale = True, use_entire_dataset = True, 
                        load_all_images = False, use_height_map = True, distraction_merge = False, 
                        distraction_merge_to_one = False, original_merge_to_one = False, 
                        molecular_imprinting_name = 'MPA'):


    data_dir = pathlib.Path(img_path)
    image_count = len(list(data_dir.glob('*.jpg')))
    logging.debug(f"image count = {image_count}")


    def sift_features_vector(src, image_path, random_keypoints_upper: int = 2500, height_map=None):
        src = rgb2gray(src)
        img_adapteq = exposure.equalize_adapthist(src, clip_limit=0.03)
        print(img_adapteq.shape)
        descriptor_extractor = SIFT()
        # descriptor_extractor = ORB(n_keypoints=50)
        descriptor_extractor.detect_and_extract(img_adapteq)
        keypoints = descriptor_extractor.keypoints
        descriptors = descriptor_extractor.descriptors
        
        # random select 100 keypoints
        random_keypoints = np.random.randint(0, len(keypoints), random_keypoints_upper) # 200 for DMMP, 500 for MP, 1000 for MP, 2600 for statics
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
        plt.savefig(f"{output_path}/{image_path}")
        plt.close()
        if height_map is not None:
            descriptors = np.hstack((descriptors, np.array(height_value).reshape(-1, 1)))
        print(descriptors.shape)
        return descriptors, keypoints



    def generate_height_map(size, offset: int = 315):
        # Define the radius of the sphere
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
                resize_image = resize_image[150:350, 150:350]
                if use_height_map:
                    height_map = generate_height_map(200)
                    resize_image = np.dstack((resize_image.astype(np.float32), height_map))
                feature_vector = sift_features_vector(resize_image, os.path.basename(path))
                x.append(feature_vector)
                y.append(path.name.split('-')[2].replace(' ', ''))
        else:
            resize_image = cv2.resize(src=src, dsize=(int(width / 2), int(height / 2)))
            # crop image 200x200
            resize_image = resize_image[50:450, 50:450]
            if use_height_map:
                height_map = generate_height_map(400)
            else:
                height_map = None
            feature_vector, keypoints = sift_features_vector(resize_image, os.path.basename(path), height_map=height_map)
            x.append(keypoints)
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
    logging.debug(x.shape)
    return x, y


class data_mining: 
    def method_1(self, x, y, color_discrete_map = None):
        # Assuming y is a list of strings corresponding to the labels of each point in x
        # Corrected color map in RGB format
        if (color_discrete_map == None):
            color_discrete_map = {
                "DMMP": 'red',
                "NaBF4": 'green',
                "KF6P": 'blue',
                "MPA": 'gold',
                "MP": 'purple'
            }

        # Calculate the center of the image
        # Assuming all images have the same size
        image_center = [200, 200]

        # Initialize lists to store all angles and radii for each class
        all_angles_classes = {label: [] for label in color_discrete_map.keys()}
        all_radii_classes = {label: [] for label in color_discrete_map.keys()}

        # Convert keypoints to polar coordinates
        for keypoints, label in zip(x, y):
            # Shift keypoints so that the center of the image is the origin
            shifted_keypoints = keypoints - image_center

            # Calculate the angles and radii
            angles = np.arctan2(shifted_keypoints[:, 0], shifted_keypoints[:, 1])
            radii = np.hypot(shifted_keypoints[:, 0], shifted_keypoints[:, 1])

            # Append the angles and radii to the lists if radius <= 200
            for angle, radius in zip(angles, radii):
                if radius <= 200:
                    all_angles_classes[label].append(angle)
                    all_radii_classes[label].append(radius)

        # Calculate the minimum number of keypoints across all classes
        min_keypoints = min(len(angles) if len(angles) != 0 else 99999 for angles in all_angles_classes.values())

        # Balance the number of keypoints for each class
        for label in all_angles_classes.keys():
            if len(all_angles_classes[label]) > min_keypoints:
                # If the number of keypoints is more than the minimum, randomly select keypoints equal to the minimum number
                indices = np.random.choice(len(all_angles_classes[label]), size=min_keypoints, replace=False)
                all_angles_classes[label] = np.array(all_angles_classes[label])[indices]
                all_radii_classes[label] = np.array(all_radii_classes[label])[indices]
        return color_discrete_map, all_angles_classes, all_radii_classes


class data_visualization:
    def __init__(self):
        logging.debug(pio.templates)
        pio.templates.default = 'plotly'


    def method_1(self, x, y, color_discrete_map = None):
        # Assuming x and y are defined properly
        # x should be a list of arrays with shape (N, 2), where N is the number of keypoints
        # y should be a list of strings corresponding to the labels of each point in x

        # Corrected color map in RGB format
        if (color_discrete_map == None):
            color_discrete_map = {
                "DMMP": [255, 0, 0],  # Red
                "NaBF4": [0, 255, 0],  # Green
                "KF6P": [0, 0, 255],  # Blue
                "MPA": [255, 165, 42],  # Goldenrod
                "MP": [128, 0, 128]  # Purple
            }

        fig = plt.figure(figsize=(10, 10))

        # Ensure x and y have the same length
        assert len(x) == len(y), "x and y must have the same length"

        # Iterate over the keypoints
        for i in range(len(x)):
            # Get the label for the current keypoint
            label = y[i]

            # Get the color for the current label
            color = color_discrete_map[label]

            # Convert color to RGB format for matplotlib
            color_rgb = [c / 255.0 for c in color]

            # Get the coordinates of the current keypoint
            coordinates = x[i]

            # Scatter plot for keypoints
            plt.scatter(coordinates[:, 1], coordinates[:, 0], facecolors='none', edgecolors=color_rgb)

        # Display the plot
        plt.show()

    def method_2(self, all_angles_classes, color_discrete_map, molecular_imprinting_name, out_dir = "./output"):
        # Plot the distribution of angles for each class
        fig = go.Figure()
        for label, angles in all_angles_classes.items():
            fig.add_trace(go.Histogram(x=angles, name=label, marker_color=color_discrete_map[label], nbinsx=15))  # Increased number of bin
        fig.update_layout(barmode='overlay', title_text='%s Distribution of Angles' % molecular_imprinting_name, xaxis_title_text='Angle (radians)', yaxis_title_text='Frequency')
        fig.update_traces(opacity=0.75)
        fig.show()
        fig.write_html(f'{out_dir}/{molecular_imprinting_name}_angle_distribution.html')

    def method_3(self, all_radii_classes, color_discrete_map, molecular_imprinting_name, out_dir = "./output"):
        # Plot the distribution of radii for each class
        fig = go.Figure()
        for label, radii in all_radii_classes.items():
            fig.add_trace(go.Histogram(x=radii, name=label, marker_color=color_discrete_map[label], nbinsx=200))  # Increased number of bin
        fig.update_layout(barmode='overlay', title_text='%s Distribution of Radii' % molecular_imprinting_name , xaxis_title_text='Radius', yaxis_title_text='Frequency')
        fig.update_traces(opacity=0.75)
        fig.show()
        fig.write_html(f'{out_dir}/{molecular_imprinting_name}_raddii_distribution.html')

    def method_4(self, all_angles_classes, color_discrete_map, molecular_imprinting_name):
        # Plot the distribution of angles for each class
        fig = go.Figure()
        for label, angles in all_angles_classes.items():
            fig.add_trace(go.Histogram(x=angles, name=label, marker_color=color_discrete_map[label], nbinsx=360, cumulative_enabled=True))  # Increased number of bin
        fig.update_layout(barmode='overlay', title_text='Cumulative Distribution of Angles', xaxis_title_text='Angle (radians)', yaxis_title_text='Cumulative Frequency')
        fig.update_traces(opacity=0.75)
        fig.show()

    def method_5(self, all_radii_classes, color_discrete_map, molecular_imprinting_name):
        # Plot the distribution of radii for each class
        fig = go.Figure()
        for label, radii in all_radii_classes.items():
            fig.add_trace(go.Histogram(x=radii, name=label, marker_color=color_discrete_map[label], nbinsx=200, cumulative_enabled=True))  # Increased number of bin
        fig.update_layout(barmode='overlay', title_text='Cumulative Distribution of Radii', xaxis_title_text='Radius', yaxis_title_text='Cumulative Frequency')
        fig.update_traces(opacity=0.75)
        fig.show()




# TODO check if this part is needed

# # Assuming y is a list of strings corresponding to the labels of each point in x

# # Initialize a dictionary to store the counts
# keypoints_counts = {label: 0 for label in set(y)}

# # Iterate over the labels
# for label in y:
#     # Increment the count for the current label
#     keypoints_counts[label] += 1

# # Print the total number of keypoints for each class
# for label, count in keypoints_counts.items():
#     print(f"Total number of keypoints for class {label}: {count}")




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
    configure_logging(DEBUG)

    molecular_imprinting_name = 'MPA'
    data_miner = data_mining()
    data_visual = data_visualization()

    x, y = image_preprocessing()
    color_discrete_map, all_angles_classes, all_radii_classes = data_miner.method_1(x, y)
    data_visual.method_1(x, y)
    data_visual.method_2(all_angles_classes, color_discrete_map, molecular_imprinting_name)
    data_visual.method_3(all_radii_classes, color_discrete_map, molecular_imprinting_name)

    data_visual.method_4(all_angles_classes, color_discrete_map, molecular_imprinting_name)
    data_visual.method_5(all_radii_classes, color_discrete_map, molecular_imprinting_name)