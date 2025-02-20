<h1 align="center">
<strong>A General Data Tool of Mining Graphic Feature Patterns from Photonic Crystal Spheres</strong>
</h1>

<p align="center">
    <a href='https://www.linkedin.com/in/hantao-zhong/' target='_blank'>Hantao Zhong†</a>&emsp;
    <a href='TODO' target='_blank'>Yumeng Gan†</a>&emsp; 
    <a href='TODO' target='_blank'>Jiecheng Cui†</a>&emsp;
    <a href='https://www.linkedin.com/in/guanlin-li/' target='_blank'>Guanlin Li‡</a>&emsp;
    <a href='TODO' target='_blank'>Changxu Lin*</a>&emsp;
    <br>
    †Equal Contribution&emsp; ‡Second author&emsp; *Corresponding author
    <!-- <br>
    Royal College of Art&emsp;University of Edinburgh&emsp;University of Cambridge&emsp;
    University College London -->
</p>





<!-- 
# photonic_crystals_pattern_mining

I will write some todo here as instructions: -->

# Environment

Clone this github repo to your local directory. 

```bash
git clone https://github.com/MrTooOldDriver/photonic_crystals_pattern_mining
```

Use conda to create a new environment. Please use environment.yml to create a new environment. 

```bash
conda create --name crystal_mining python=3.10
```

Install the package. 

```bash
pip install ./package/dist/photonic_crystals_pattern_mining-0.1.0-py3-none-any.whl
```

# Dataset Preparation

You can use dataset from this repoistory or prepare your own dataset. Images must be in the '*.jpg' format and located in a subdirectory. 

Here's an example dataset folder structure: 

```
dataset
├─ M-DMMP-NaBF4-10-6M
│  ├─ M-DMMP-NaBF4-10-6M-1.jpg
│  ├─ M-DMMP-NaBF4-10-6M-2.jpg
│  ├─ M-DMMP-NaBF4-10-6M-3.jpg
│  ├─ ...
├─ M-DMMP-DMMP-10-3M
│  ├─ M-DMMP-DMMP-10-3M-1.jpg
│  ├─ M-DMMP-DMMP-10-3M-2.jpg
│  ├─ M-DMMP-DMMP-10-3M-3.jpg
│  ├─ M-DMMP-DMMP-10-3M-4.jpg
│  ├─ ...
├─ ...
```

Note that the names of subfolders and images are irrelevant; however, the top folder's must be named `dataset`. 

# Crystal Detection

The first step is to detect all crystal from raw images. Raw images should be located under folder `dataset` following the file structure mentioned above. All outputs should be generated into the `output` folder. 

Please refactor all three files:
[IRCS_cv_locator_new_data_temporal.py](./src/IRCS_cv_locator_new_data_temporal.py) [IRCS_cv_locator_new_data.py](./src/IRCS_cv_locator_new_data.py) [IRCS_cv_locator.py](./src/IRCS_cv_locator.py), and the example usecase: [example_usecase.ipynb](./example_usecase.ipynb). 



# Feature extration

The major part of this project is to using detected images from /output then do all the processing。
Please refer to the example usecase: [example_usecase.ipynb](./example_usecase.ipynb). 

IRCS_CANNY_TSNE, IRCS_HOG_TSNE, and IRCS_SIFT_TSNE are image feature extraction package with visulization.

IRCS_SIFT_PATTERN, IRCS_SIFT_PATTERN_DENSITY_MAP, and IRCS_SIFT_PATTERN_ALIGNMENT are more of additional visulization of SIFT. Please use these packages after you run IRCS_SIFT_TSNE. 



