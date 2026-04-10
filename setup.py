from setuptools import setup, find_packages

setup(
    name="fake_news_detector",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "nltk",
        "scikit-learn",
        "pymongo",
        "mlflow",
        "dagshub",
        "dvc",
        "python-dotenv",
        "pyyaml",
        "tqdm",
    ],
)