from setuptools import setup, find_packages
from pathlib import Path

here = Path(__file__).parent.resolve()

# Load README
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="biomedical_classifier",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@domain.com",
    description="Multimodal Biomedical Signal Classification using Hybrid CNN-LSTM with Adaptive Normalization, Attention, and LRP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/biomedical-classifier",
    packages=find_packages(exclude=["tests", "docs"]),
    python_requires=">=3.11",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "torchaudio>=2.1.0",
        "numpy>=1.25.0",
        "scipy>=1.11.0",
        "pandas>=2.1.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.12.3",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "tensorboard>=2.15.0",
        "tensorboardX>=2.7",
        "jupyter>=1.0.0",
        "notebook>=7.0.0",
        "pillow>=10.0.0",
        "opencv-python>=4.9.0",
        "mne>=1.4.0",
        "wfdb>=4.0.0",
        "einops>=0.9.0",
        "pytorch-lightning>=2.3.0",
        "wandb>=0.15.0",
        "shap>=0.42.0",
        "captum>=0.7.0",
        "opencv-contrib-python>=4.9.0",
        "scikit-image>=1.22.0",
        "pyvista>=0.40.0",
        "plotly>=6.1.0",
        "torchmetrics>=0.11.4"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics"
    ],
    entry_points={
        "console_scripts": [
            "train=bin.train:main",
            "evaluate=bin.evaluate:main",
            "explain=bin.explain:main"
        ]
    }
)
