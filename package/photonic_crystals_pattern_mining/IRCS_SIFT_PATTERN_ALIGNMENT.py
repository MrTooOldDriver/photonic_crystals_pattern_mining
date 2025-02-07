import pathlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import sys
import plotly.io as pio
import plotly.graph_objects as go
import matplotlib.image as mpimg
from skimage.feature import SIFT
from skimage import exposure
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_filter1d
from scipy.fft import fft, fftfreq

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


def image_preprocessing(img_path: str = './output/rgb', output_path: str = './output/sift', 
                        rotation_aug = False, experiment_with_gray_scale = True, use_entire_dataset = True, 
                        load_all_images = False, use_height_map = True, distraction_merge = False, 
                        distraction_merge_to_one = False, original_merge_to_one = False, 
                        molecular_imprinting_name = 'DMMP', random_keypoints_upper: int = 2500, offset: int = 315):
    """
    _summary_

    Args:
        img_path (str, optional): _description_. Defaults to './output/rgb'.
        output_path (str, optional): _description_. Defaults to './output/sift'.
    
    Returns: 
        x, y
    """

    def sift_features_vector(src, image_path, height_map=None):
        src = rgb2gray(src)
        img_adapteq = exposure.equalize_adapthist(src, clip_limit=0.03)
        print(img_adapteq.shape)
        descriptor_extractor = SIFT()
        # descriptor_extractor = ORB(n_keypoints=50)
        descriptor_extractor.detect_and_extract(img_adapteq)
        keypoints = descriptor_extractor.keypoints
        descriptors = descriptor_extractor.descriptors
        
        # random select 100 keypoints
        random_keypoints = np.random.randint(0, len(keypoints), random_keypoints_upper) # 200 for DMMP, 500 for MP, 1000 for MP, 2500 for statics
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


    def generate_height_map(size):
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



    # get all files ending with .jpg
    data_dir = pathlib.Path(img_path)
    image_count = len(list(data_dir.glob('*.jpg')))
    logging.debug(f"image count = {image_count}")

    # # visualise the height map in 3D TODO
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

    # Load the image, seems test case
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





# x, y = image_preprocessing()


# molecular_imprinting_name = 'DMMP'





# visualization TODO
# # %%

# # Assuming x and y are defined properly
# # x should be a list of arrays with shape (N, 2), where N is the number of keypoints
# # y should be a list of strings corresponding to the labels of each point in x

# # Corrected color map in RGB format
# color_discrete_map = {
#     "DMMP": [255, 0, 0],  # Red
#     "NaBF4": [0, 255, 0],  # Green
#     "KF6P": [0, 0, 255],  # Blue
#     "MPA": [255, 165, 42],  # Goldenrod
#     "MP": [128, 0, 128]  # Purple
# }

# fig = plt.figure(figsize=(10, 10))

# # Ensure x and y have the same length
# assert len(x) == len(y), "x and y must have the same length"

# # Iterate over the keypoints
# for i in range(len(x)):
#     # Get the label for the current keypoint
#     label = y[i]

#     # Get the color for the current label
#     color = color_discrete_map[label]

#     # Convert color to RGB format for matplotlib
#     color_rgb = [c / 255.0 for c in color]

#     # Get the coordinates of the current keypoint
#     coordinates = x[i]

#     # Scatter plot for keypoints
#     plt.scatter(coordinates[:, 1], coordinates[:, 0], facecolors='none', edgecolors=color_rgb)

# # Display the plot
# plt.show()









# # data check? TODO
# # %%
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



# data mining

class data_mining:
    def method_1(self, x, y, color_discrete_map: map = None):
        logging.debug(pio.templates)
        pio.templates.default = 'plotly'

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

        angle_bin = 20
        radii_bin = 200

        # Initialize lists to store all angles and radii for each class
        all_angles_classes = {label: [] for label in color_discrete_map.keys()}
        all_radii_classes = {label: [] for label in color_discrete_map.keys()}

        # Convert keypoints to polar coordinates
        for keypoints, label in zip(x, y):
            # Shift keypoints so that the center of the image is the origin
            shifted_keypoints = keypoints - image_center

            # Calculate the angles and radii
            angles = np.arctan2(shifted_keypoints[:, 0], shifted_keypoints[:, 1])
            angles_in_degree = np.rad2deg(angles)
            radii = np.hypot(shifted_keypoints[:, 0], shifted_keypoints[:, 1])
            
            hist, edges = np.histogram(angles_in_degree, bins=angle_bin)
            angles_hist_min_index = np.argmin(hist)
            offset = edges[angles_hist_min_index]
            print('alightment found = %s' % offset)
            angles_in_degree = angles_in_degree + offset
            
            angle = np.deg2rad(angles_in_degree)
            
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
        
        return all_angles_classes, angle_bin, all_radii_classes, radii_bin, color_discrete_map
    
    def method_2(self, x, y, color_discrete_map: map = None):
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

        angle_bin = 45

        # Initialize lists to store all angles for each class
        all_angles_classes = {label: [] for label in color_discrete_map.keys()}

        # Convert keypoints to polar coordinates
        for keypoints, label in zip(x, y):
            logging.debug(label)
            # Shift keypoints so that the center of the image is the origin
            shifted_keypoints = keypoints - image_center

            # Calculate the angles
            angles = np.arctan2(shifted_keypoints[:, 0], shifted_keypoints[:, 1])
            angles_in_degree = np.rad2deg(angles)

            hist, edges = np.histogram(angles_in_degree, bins=angle_bin)
            angles_hist_min_index = np.argmin(hist)
            offset = edges[angles_hist_min_index]
            print('alignment found = %s' % offset)
            angles_in_degree = angles_in_degree + offset
            # convert all angle into -180 and 180 range
            angles_in_degree = (angles_in_degree + 180) % 360 - 180
            angle = np.deg2rad(angles_in_degree)

            # Append the angles to the lists
            all_angles_classes[label].extend(angle)

        to_del = []
        for key, value in all_angles_classes.items():
            if sum(value) == 0:
                to_del.append(key)
        for key in to_del:
            del all_angles_classes[key]
        return all_angles_classes, angle_bin, color_discrete_map
        

    def method_3(self, x, y, color_discrete_map: map = None):
        # plot line graph for radii

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
        image_center = [200, 200]

        radii_bin = 30

        # Initialize lists to store all radii for each class
        all_radii_classes = {label: [] for label in color_discrete_map.keys()}

        # Convert keypoints to polar coordinates
        for keypoints, label in zip(x, y):
            # Shift keypoints so that the center of the image is the origin
            shifted_keypoints = keypoints - image_center

            # Calculate the radii
            radii = np.hypot(shifted_keypoints[:, 0], shifted_keypoints[:, 1])

            # Append the radii to the lists if radius <= 200
            for radius in radii:
                if radius <= 200:
                    all_radii_classes[label].append(radius)

        return all_radii_classes, radii_bin, color_discrete_map

    def method_4(self, x, y, color_discrete_map = None):
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

        angle_bin = 45

        # Initialize lists to store all angles for each class
        all_angles_classes = {label: [] for label in color_discrete_map.keys()}

        # Convert keypoints to polar coordinates
        for keypoints, label in zip(x, y):
            # Shift keypoints so that the center of the image is the origin
            shifted_keypoints = keypoints - image_center

            # Calculate the angles
            angles = np.arctan2(shifted_keypoints[:, 0], shifted_keypoints[:, 1])
            angles_in_degree = np.rad2deg(angles)

            hist, edges = np.histogram(angles_in_degree, bins=angle_bin)
            angles_hist_min_index = np.argmin(hist)
            offset = edges[angles_hist_min_index]
            angles_in_degree = angles_in_degree + offset
            # Convert all angles to the range [-180, 180]
            angles_in_degree = (angles_in_degree + 180) % 360 - 180
            angle = np.deg2rad(angles_in_degree)

            # Append the angles to the lists
            all_angles_classes[label].extend(angle)

        to_del = []
        for key, value in all_angles_classes.items():
            if sum(value) == 0:
                to_del.append(key)
        for key in to_del:
            del all_angles_classes[key]
        
        return all_angles_classes, angle_bin, color_discrete_map





class data_visualization:
    def method_1(self, all_angles_classes, angle_bin, all_radii_classes, radii_bin, color_discrete_map, molecular_imprinting_name):
        # visualization
        # Plot the distribution of angles for each class
        fig = go.Figure()
        for label, angles in all_angles_classes.items():
            fig.add_trace(go.Histogram(x=angles, name=label, marker_color=color_discrete_map[label], nbinsx=angle_bin))  # Increased number of bin
        fig.update_layout(barmode='overlay', title_text='%s Distribution of Angles' % molecular_imprinting_name, xaxis_title_text='Angle (radians)', yaxis_title_text='Frequency')
        fig.update_traces(opacity=0.75)
        fig.show()

        # Plot the distribution of radii for each class
        fig = go.Figure()
        for label, radii in all_radii_classes.items():
            fig.add_trace(go.Histogram(x=radii, name=label, marker_color=color_discrete_map[label], nbinsx=radii_bin))  # Increased number of bin
        fig.update_layout(barmode='overlay', title_text='%s Distribution of Radii' % molecular_imprinting_name , xaxis_title_text='Radius', yaxis_title_text='Frequency')
        fig.update_traces(opacity=0.75)
        fig.show()

    def method_2(self, all_angles_classes, angle_bin, color_discrete_map, molecular_imprinting_name):
        # Plot the distribution of angles for each class as a smoothed line graph
        fig = go.Figure()
        for label, angles in all_angles_classes.items():
            print(label)
            hist, bin_edges = np.histogram(angles, bins=angle_bin)
            total_sum = np.sum(hist)
            hist = hist / total_sum
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            smoothed_hist = gaussian_filter1d(hist, sigma=0.01) # 0.5  # Apply Gaussian smoothing
            fig.add_trace(go.Scatter(x=bin_centers, y=smoothed_hist, mode='lines', name=label, line=dict(color=color_discrete_map[label])))

        fig.update_layout(title_text='', xaxis_title_text='Angle (radians)', yaxis_title_text='Frequency', showlegend=False)

        fig.update_layout(autosize=False,
                            scene_camera_eye=dict(x=1, y=1, z=2),
                            width=500, height=500,
                            margin=dict(l=20, r=20, b=20, t=20)
            )


        fig.show()

        fig.write_image("output_distribution/%s-line.png" % molecular_imprinting_name, height=500, width=500)

    def method_3(self, all_radii_classes, radii_bin, color_discrete_map, molecular_imprinting_name):
        # Plot the distribution of radii for each class as a smoothed line graph
        fig = go.Figure()
        for label, radii in all_radii_classes.items():
            hist, bin_edges = np.histogram(radii, bins=radii_bin)
            total_sum = np.sum(hist)
            hist = hist / total_sum
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            smoothed_hist = gaussian_filter1d(hist, sigma=0.01)  # Apply Gaussian smoothing
            fig.add_trace(go.Scatter(x=bin_centers, y=smoothed_hist, mode='lines', name=label, line=dict(color=color_discrete_map[label])))

        fig.update_layout(title_text='', xaxis_title_text='Radius', yaxis_title_text='Frequency',showlegend=False)

        fig.update_layout(autosize=False,
                            scene_camera_eye=dict(x=1, y=1, z=2),
                            width=500, height=500,
                            margin=dict(l=20, r=20, b=20, t=20)
            )

        fig.show()

        fig.write_image("output_distribution/%s-radii-line.png" % molecular_imprinting_name, height=500, width=500)

    def method_4(self, all_angles_classes, angle_bin, color_discrete_map, molecular_imprinting_name):

        # Apply Fourier Transform to the smoothed data and visualize the result
        fig_fft = go.Figure()
        for label, angles in all_angles_classes.items():
            hist, bin_edges = np.histogram(angles, bins=angle_bin)
            total_sum = np.sum(hist)
            hist = hist / total_sum 
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            smoothed_hist = gaussian_filter1d(hist, sigma=0.01)  # Apply Gaussian smoothing

            # Apply Fourier Transform
            yf = fft(smoothed_hist)
            xf = fftfreq(len(smoothed_hist), bin_centers[1] - bin_centers[0])
            
            # sort the frequency and amplitude from low to high
            yf = yf[np.argsort(xf)]
            xf = np.sort(xf)
            
            # filter out the negative frequency
            yf = yf[xf > 0]
            xf = xf[xf > 0]
            
            fig_fft.add_trace(go.Scatter(x=xf, y=np.abs(yf), mode='lines', name=label, marker=dict(color=color_discrete_map[label])))

        fig_fft.update_layout(title_text='', xaxis_title_text='Frequency', yaxis_title_text='Amplitude', showlegend=False)

        fig_fft.update_layout(autosize=False,
                            scene_camera_eye=dict(x=1, y=1, z=2),
                            width=500, height=500,
                            margin=dict(l=20, r=20, b=20, t=20)
            )

        fig_fft.show()

        fig_fft.write_image("output_distribution/%s-fourier.png" % molecular_imprinting_name, height=500, width=500)

    def method_5(self, img_dir: str = "./output_distribution", save_path: str = './output_distribution/merged_angle_image.png'):
        # List of image names
        image_names = [
            'DMMP-line', 'MP-line',
            'MPA-line', 'DMMP-fourier',
            'MP-fourier', 'MPA-fourier', 
        ]

        # Create a figure with a 3x3 grid layout
        fig, axes = plt.subplots(2, 3, figsize=(50, 40))

        # Loop through the image names and add each image to the grid
        for i, ax in enumerate(axes.flat):
            img_path = f'{img_dir}/{image_names[i]}.png'
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.axis('off')  # Remove axis labels and ticks

        # Set the background color to white
        fig.patch.set_facecolor('white')

        # Save the merged image
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()

    def method_6(self, img_dir: str = "./output_distribution", save_path: str = './output_distribution/merged_radii_image.png'):
        # List of image names
        image_names = [
            'DMMP-radii-line', 'MP-radii-line',
            'MPA-radii-line', 
        ]

        # Create a figure with a 3x3 grid layout
        fig, axes = plt.subplots(1, 3, figsize=(50, 20))

        # Loop through the image names and add each image to the grid
        for i, ax in enumerate(axes.flat):
            img_path = f'{img_dir}/{image_names[i]}.png'
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.axis('off')  # Remove axis labels and ticks

        # Set the background color to white
        fig.patch.set_facecolor('white')

        # Save the merged image
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()




# TODO
# # Apply Fourier Transform to the smoothed data and print the result
# for label, angles in all_angles_classes.items():
#     hist, bin_edges = np.histogram(angles, bins=angle_bin)
#     bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
#     smoothed_hist = gaussian_filter1d(hist, sigma=0.01)  # Apply Gaussian smoothing

#     # Apply Fourier Transform
#     yf = fft(smoothed_hist)
#     xf = fftfreq(len(smoothed_hist), bin_centers[1] - bin_centers[0])

#     print(f"Fourier Transform result for {label}:")
#     print("Frequencies:", xf)
#     print("Amplitudes:", np.abs(yf))




# EXAMPLE USE CASE
if __name__ == "__main__":
    molecular_imprinting_name = 'DMMP'
    data_miner = data_mining()
    data_visual = data_visualization()

    x, y = image_preprocessing()

    all_angles_classes, angle_bin, all_radii_classes, radii_bin, color_discrete_map = data_miner.method_1(x, y)
    data_visual.method_1(all_angles_classes, angle_bin, all_radii_classes, radii_bin, color_discrete_map, molecular_imprinting_name)

    all_angles_classes, angle_bin, color_discrete_map = data_miner.method_2(x, y)
    data_visual.method_2(all_angles_classes, angle_bin, color_discrete_map, molecular_imprinting_name)

    all_radii_classes, radii_bin, color_discrete_map = data_miner.method_3(x, y)
    data_visual.method_3(all_radii_classes, radii_bin, color_discrete_map, molecular_imprinting_name)

    all_radii_classes, radii_bin, color_discrete_map = data_miner.method_4(x, y)
    data_visual.method_4(all_angles_classes, angle_bin, color_discrete_map, molecular_imprinting_name)

    data_visual.method_5()

    data_visual.method_6()