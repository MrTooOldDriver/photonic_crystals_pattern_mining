import pathlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import plotly.io as pio
import plotly.express as px
from sklearn.mixture import GaussianMixture
from skimage.feature import hog
from skimage.feature import canny
from skimage.feature import SIFT
from sklearn.model_selection import train_test_split
from matplotlib.colors import Normalize
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from skimage import exposure
from sklearn import preprocessing as p
from skimage.color import rgb2gray

# data_dir = pathlib.Path('./output_new_data_temporal/rgb')
data_dir = pathlib.Path('./output/output_new_data_temporal_no_radius_adjust/rgb')
image_count = len(list(data_dir.glob('*.jpg')))
print(image_count)





def hog_feature_vector(src, pixels_per_cell):
    fd = hog(src, orientations=9, pixels_per_cell=pixels_per_cell, cells_per_block=(1, 1), channel_axis=2)
    # fd = hog(src, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), channel_axis=2)
    return fd

def canny_feature_vector(src):
    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges.flatten() / 255.0

def sift_features_vector(src, random_keypoints_upper: int = 1000, height_map=None):
    src = rgb2gray(src)
    img_adapteq = exposure.equalize_adapthist(src, clip_limit=0.03)
    # print(img_adapteq.shape)
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
    
    if height_map is not None:
        descriptors = np.hstack((descriptors, np.array(height_value).reshape(-1, 1)))
    # print(descriptors.shape)
    return descriptors

def generate_height_map(size, offset: int = 315):
        size = size + 2*offset

        r = 1.0

        # Create a 2D grid of x and y coordinates
        x, y = np.meshgrid(np.linspace(-r, r, size), np.linspace(-r, r, size))

        # Calculate the corresponding z coordinates
        # Note: For points outside the sphere, this will be NaN
        z_square = np.clip(r**2 - x**2 - y**2, 0, None)
        z = np.sqrt(z_square)

        # We set points outside the sphere to zero height for visualization
        z[np.isnan(z)] = 0

        z = z[offset:size-offset, offset:size-offset]

        return z

# Enhance local contrast by boosting small bright and dark features.
def image_enhance(src):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    # Top Hat Transform
    topHat = cv2.morphologyEx(src, cv2.MORPH_TOPHAT, kernel)
    # Black Hat Transform
    blackHat = cv2.morphologyEx(src, cv2.MORPH_BLACKHAT, kernel)
    src = src + topHat - blackHat
    # print('enhance on')
    return src


experiment_with_gray_scale = True
use_entire_dataset = True
enhance = True


# type_of_analysis = "HOG_before_diff"
# type_of_analysis = "HOG_after_diff"
type_of_analysis = "CANNY"
# type_of_analysis = "SIFT"


# pixels_per_cell = (10, 10)
# pixels_per_cell = (20, 20)
# pixels_per_cell = (50, 50)
pixels_per_cell = (100, 100)
# pixels_per_cell = (125, 125)

# Data selection
molecular_imprinting_name, load_all_images = 'all', True
# molecular_imprinting_name = 'C4'
# molecular_imprinting_name = 'C6'
# molecular_imprinting_name = 'C8'

image_limit = {
        'C4': 12,
        'C6': 18,
        'C8': 16,
        'all': 12,
    }

images_cache = {}
for path in sorted(data_dir.glob('*.jpg')):
    split_name = path.name.split('-')
    solution_name = split_name[0]
    ion_name = split_name[2]
    sequence_name = split_name[-2]
    frame_name = split_name[-1].split('.')[0]
    if not load_all_images and solution_name != molecular_imprinting_name:  # False
        continue
    
    src = cv2.imread(str(path))
    if experiment_with_gray_scale:      # True
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        src = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)
    else:
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)


    height, width = src.shape[:2]
    center = (width / 2, height / 2)
    if enhance:     # True
        src = image_enhance(src)
    
    identity = solution_name + '-' + ion_name + '-' + sequence_name
    if identity not in images_cache:
        images_cache[identity] = {}
    if int(frame_name) < image_limit[molecular_imprinting_name]:
        images_cache[identity][frame_name] = src



x = []
y = []


for identity, frames in images_cache.items():
    # For every set in images_cache, get the subdictionary and sort by keys (frame name)
    sorted_frames = sorted(frames.keys(), key=lambda x: int(x))
    feature_vector = []



    for i in range(len(sorted_frames) - 1):
        if (type_of_analysis == "HOG_after_diff"):
            diff_image = cv2.absdiff(frames[sorted_frames[i]], frames[sorted_frames[i + 1]])
            feature_vector.append(hog_feature_vector(diff_image, pixels_per_cell))
        elif (type_of_analysis == "HOG_before_diff"):
            img1 = hog_feature_vector(frames[sorted_frames[i]], pixels_per_cell)
            img2 = hog_feature_vector(frames[sorted_frames[i + 1]], pixels_per_cell)
            diff_image = cv2.absdiff(img1, img2)
            feature_vector.append(diff_image)
        elif (type_of_analysis == "CANNY"):
            diff_image = cv2.absdiff(frames[sorted_frames[i]], frames[sorted_frames[i + 1]])
            feature_vector.append(canny_feature_vector(diff_image))
        elif (type_of_analysis == "SIFT"):
            diff_image = cv2.absdiff(frames[sorted_frames[i]], frames[sorted_frames[i + 1]])
            feature_vector.append(sift_features_vector(diff_image))
        else:
            raise KeyError("ERR: incorrect type_of_analysis value")

    assert len(feature_vector) == image_limit[molecular_imprinting_name] - 1
    
    feature_vector = np.array(feature_vector)
    feature_vector = feature_vector.reshape(-1)
    
    if load_all_images:     # True
        label_name = identity.split('-')[0] + identity.split('-')[1]
    else:
        label_name = identity.split('-')[1]
        
    # filter
    if label_name[-2:] == 'Br':
        continue
    if label_name[-4:] == 'Tf2N':
        continue
    
    x.append(feature_vector)
    y.append(label_name)



x = np.array(x)
y = np.array(y)
print('data loaded x=%i' % (len(x)))
print('data loaded y=%i' % (len(y)))
print('x.shape=%s' % (str(x.shape)))
print('y.shape=%s' % (str(y.shape)))



# Example HOG image

# Select an identity
identity = list(images_cache.keys())[0]

# Get the sorted frame keys
sorted_frames = sorted(images_cache[identity].keys(), key=lambda x: int(x))

# Calculate the difference between the first two frames
diff_image = cv2.absdiff(images_cache[identity][sorted_frames[0]], images_cache[identity][sorted_frames[1]])

# Apply the HOG feature extraction
fd, hog_image = hog(diff_image, orientations=9, pixels_per_cell=pixels_per_cell, cells_per_block=(1, 1), visualize=True, channel_axis=2)

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 1))

# Visualize the original difference image and the HOG image
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

ax1.axis('off')
ax1.imshow(cv2.cvtColor(diff_image, cv2.COLOR_BGR2RGB))
ax1.set_title('Difference Image')

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')

plt.show()



fd, hog_image = hog(diff_image, orientations=8, pixels_per_cell=pixels_per_cell, cells_per_block=(1, 1), visualize=True, channel_axis=2, feature_vector=False)

print(fd.shape)

fig = plt.figure(figsize=(15, 15))
# plt.axis('off')
x_loc_array = []
y_loc_array = []
unit_vector_x_array = []
unit_vector_y_array = []
colors_array = []
colormap = cm.inferno
for i in range(10):
    for j in range(10):
        feature = fd[i][j][0][0]
        x_loc = 50 + i * 100
        y_loc = 50 + j * 100
        final_radian = 0
        sum_feat = 0
        for k in range(len(feature)):
            angle_in_radian = (np.pi / 9) * k
            final_radian += angle_in_radian * feature[k]
            sum_feat += feature[k]

        unit_vector_x = np.cos(final_radian)
        unit_vector_y = np.sin(final_radian)
        x_loc_array.append(x_loc)
        y_loc_array.append(y_loc)
        unit_vector_x_array.append(unit_vector_x)
        unit_vector_y_array.append(unit_vector_y)
        colors_array.append(sum_feat)

norm = Normalize()
norm.autoscale(colors_array)

plt.quiver(x_loc_array, y_loc_array, unit_vector_x_array, unit_vector_y_array, pivot='middle', color=colormap(norm(colors_array)))
plt.imshow(diff_image)

# %%
plt.imshow(images_cache[identity][sorted_frames[0]])

# %%
plt.imshow(images_cache[identity][sorted_frames[1]])

# %%
# import matplotlib.pyplot as plt
# 
# # Calculate the total number of subplots needed
# total_subplots = sum(len(frames) - 1 for frames in images_cache.values())
# 
# # Create a figure with that many subplots arranged in a grid
# fig, axs = plt.subplots(total_subplots, 3, figsize=(12, 4 * total_subplots))
# 
# # Initialize the subplot index
# subplot_idx = 0
# 
# # Loop over each identity in the images_cache
# for identity, frames in images_cache.items():
#     # Get the sorted frame keys
#     sorted_frames = sorted(frames.keys(), key=lambda x: int(x))
# 
#     # Loop over the sorted frames from 0 to n-1
#     for i in range(len(sorted_frames) - 1):
#         # Calculate the difference between the current frame and the next frame
#         diff_image = cv2.absdiff(frames[sorted_frames[i]], frames[sorted_frames[i + 1]])
# 
#         # Plot the original images and their difference in the next subplot
#         axs[subplot_idx, 0].axis('off')
#         axs[subplot_idx, 0].imshow(cv2.cvtColor(frames[sorted_frames[i]], cv2.COLOR_BGR2RGB))
#         axs[subplot_idx, 0].set_title(f'Original Image {sorted_frames[i]}')
# 
#         axs[subplot_idx, 1].axis('off')
#         axs[subplot_idx, 1].imshow(cv2.cvtColor(frames[sorted_frames[i + 1]], cv2.COLOR_BGR2RGB))
#         axs[subplot_idx, 1].set_title(f'Original Image {sorted_frames[i + 1]}')
# 
#         axs[subplot_idx, 2].axis('off')
#         axs[subplot_idx, 2].imshow(cv2.cvtColor(diff_image, cv2.COLOR_BGR2RGB))
#         axs[subplot_idx, 2].set_title('Difference Image')
# 
#         # Increment the subplot index
#         subplot_idx += 1
# 
# # Display the figure
# plt.tight_layout()
# plt.show()

# %% [markdown]
# DATA MINING

# %%
perplexity=3
pca = TSNE(n_components=3, learning_rate='auto', init='pca', perplexity=perplexity)
x_train_pca = None
y_train = None
if use_entire_dataset:      # True
    x_train_pca = pca.fit_transform(x)
    y_train = y
# x_train_pca

# %%
n_components=3
df = pd.DataFrame(x_train_pca, columns=[f"PC{i + 1}" for i in range(n_components)])
label_list = []

for i in range(len(x_train_pca)):
    label_list.append(y_train[i])
df['label'] = label_list

# %%

print(pio.templates)
pio.templates.default = 'plotly'

# %%

fig = px.scatter_3d(df, x='PC1', y='PC2',z='PC3',  color='label', symbol='label', title='Hog TSNE %s Granularity=%s Sperplexity=%s' % (molecular_imprinting_name, str((int(1000 / pixels_per_cell[0]), int(1000 / pixels_per_cell[0]))), str(perplexity)), color_discrete_map={
                "DMMP": "red",
                "NaBF4": "green",
                "KF6P": "blue",
                "MPA": "goldenrod",
                "MP": "purple"},
                symbol_sequence= ['circle', 'circle', 'circle', 'circle'])
fig.show()

# %%
# Define the output path
output_path = data_dir.parent / f'temporal_{molecular_imprinting_name}_{type_of_analysis}_output.html'

fig.write_html(str(output_path))


