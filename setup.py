from setuptools import find_packages, setup  # type: ignore

VERSION = "0.1.0"

extras = {}
extras["quality"] = ["ruff>=0.0.241"]

setup(
    name="xlora",
    version=VERSION,
    description="X-LoRA: Mixture of LoRA Experts",
    license_files=["LICENSE"],
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="deep learning",
    license="Apache",
    author="Eric Buehler",
    author_email="ericlbuehler@gmail.com",
    url="https://github.com/EricLBuehler/xlora",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"xlora": ["py.typed"]},
    entry_points={},
    python_requires=">=3.7.0",
    install_requires=["peft", "transformers"],
    extras_require=extras,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
