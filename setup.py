from setuptools import find_packages, setup

VERSION = "0.1.0"

extras = {}
extras["quality"] = ["ruff>=0.0.241"]

setup(
    name="mole",
    version=VERSION,
    description="Mixture of LoRA Experts (MoLE)",
    license_files=["LICENSE"],
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="deep learning",
    license="Apache",
    author="EricL uehler",
    author_email="ericlbuehler@gmail.com",
    url="https://github.com/EricLBuehler/mole",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"mole": ["py.typed"]},
    entry_points={},
    python_requires=">=3.8.0",
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
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

# Release checklist
# 1. Change the version in __init__.py and setup.py to the release version, e.g. from "0.6.0.dev0" to "0.6.0"
# 2. Check if there are any deprecations that need to be addressed for this release by searching for "# TODO" in the code
# 3. Commit these changes with the message: "Release: VERSION", create a PR and merge it.
# 4. Add a tag in git to mark the release: "git tag -a VERSION -m 'Adds tag VERSION for pypi' "
#    Push the tag to git:
#      git push --tags origin main
#    It is necessary to work on the original repository, not on a fork.
# 5. Run the following commands in the top-level directory:
#      python setup.py bdist_wheel
#      python setup.py sdist
#    Ensure that you are on the clean and up-to-date main branch (git status --untracked-files=no should not list any
#    files and show the main branch)
# 6. Upload the package to the pypi test server first:
#      twine upload dist/* -r pypitest
# 7. Check that you can install it in a virtualenv by running:
#      pip install -i https://testpypi.python.org/pypi --extra-index-url https://pypi.org/simple peft
# 8. Upload the final version to actual pypi:
#      twine upload dist/* -r pypi
# 9. Add release notes to the tag on https://github.com/huggingface/peft/releases once everything is looking hunky-dory.
#      Check the notes here: https://docs.google.com/document/d/1k-sOIfykuKjWcOIALqjhFKz4amFEp-myeJUJEzNgjoU/edit?usp=sharing
# 10. Update the version in __init__.py, setup.py to the bumped minor version + ".dev0" (e.g. from "0.6.0" to "0.7.0.dev0")
