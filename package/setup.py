from setuptools import setup, find_packages

setup(
    name="photonic_crystals_pattern_mining",
    version="0.1.0",
    packages=find_packages(),
    author="test",
    author_email="test@something.com",
    description="A simple Python package test",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8", 
    install_requires = [
        "opencv-python",   # Provides cv2
        "numpy",
        "pandas",
        "matplotlib",
        "plotly",
        "scikit-learn",
        "scikit-image",
        "scipy",
        "kaleido==0.1.0"
    ]


)
