from setuptools import setup, find_packages

setup(
    name="contre",
    packages=find_packages(),
    author="Lena Feld",
    author_email="lena.feld@student.kit.edu",
    description="CONTinuum REweighting for Belle and Belle II",
    install_requires=[
        "b2luigi",
        "pandas",
        "root-pandas",
        "scikit-learn",
        "numpy"
        ],
    license='MIT',
    )
