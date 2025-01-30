# photonic_crystals_pattern_mining

I will write some todo here as instructions:

# Crystal Detection

The first step is to detect all crystal from raw images. raw images located under /dataset. All the outputs is under /output

Please refactor all three files:
[IRCS_cv_locator_new_data_temporal.py](IRCS_cv_locator_new_data_temporal.py) [IRCS_cv_locator_new_data.py](IRCS_cv_locator_new_data.py) [IRCS_cv_locator.py](IRCS_cv_locator.py)

# Feature extration

The major part of this project is to using detected images from /output then do all the processing。
Please refer to all ipynb nootbook

[IRCS_CANNY_TSNE.ipynb](IRCS_CANNY_TSNE.ipynb) [IRCS_HOG_TSNE.ipynb](IRCS_HOG_TSNE.ipynb) [IRCS_SIFT_TSNE.ipynb](IRCS_SIFT_TSNE.ipynb)
Those 3 ipynb are top priotiry to be completed with. They are image feature extraction with visulization.

[IRCS_SIFT_PATTERN.ipynb](IRCS_SIFT_PATTERN.ipynb) [IRCS_SIFT_PATTERN_DENSITY_MAP.ipynb](IRCS_SIFT_PATTERN_DENSITY_MAP.ipynb) [IRCS_SIFT_PATTERN_ALIGNMENT.ipynb](IRCS_SIFT_PATTERN_ALIGNMENT.ipynb)
Those 3 ipynb are more of additional visulization of SIFT. Do this three later.