from setuptools import setup, find_packages
from woe_scoring import __version__

DISTNAME = "woe_scoring"
DESCRIPTION = "Weight Of Evidence Transformer and LogisticRegression model with scikit-learn API"

with open("README.md") as f:
    LONG_DESCRIPTION = f.read()

MAINTAINER = "Stroganov Kirill"
MAINTAINER_EMAIL = "kiraplenkin@gmail.com"
URL = "https://github.com/kiraplenkin"
DOWNLOAD_URL = "https://pypi.org/project/woe-scoring/#files"
LICENSE = "MIT"

setup(
    name=DISTNAME,
    version=__version__,
    description="LogisticRegression model on Weight Of Evidence transformated variables",
    long_description='',
    author=MAINTAINER,
    author_email=MAINTAINER_EMAIL,
    url=URL,
    download_url="https://github.com/kiraplenkin/woe_scoring/archive/refs/tags/v0.1.5.tar.gz",
    license='',
    packages=find_packages(),
    include_package_data=True,
    keywords=[
        "WOE",
        "Weight Of Evidence",
        "Monotone Weight Of Evidence Transformation",
        "Scorecard",
        "LogisticRegression"
    ],
    install_requires=["numpy", "pandas", "scikit-learn", "statsmodels", "scipy"],
    zip_safe=False
)
