from setuptools import setup, find_packages

setup(
    name="sentiment_classification",
    version="0.1.0",
    description="Lao Sentiment Analysis with PyTorch and HuggingFace",
    author="Your Name/Your Team",
    packages=find_packages(where="src"), # Tìm các package (thư mục chứa __init__.py) nằm trong /src
    package_dir={"": "src"},             # Chỉ định thư mục gốc của package
    install_requires=[
        "torch",
        "transformers",
        "datasets",
        "evaluate",
        "scikit-learn",
        "pandas",
        "numpy",
        "peft",
        "accelerate"
    ],
    python_requires=">=3.8",
)
