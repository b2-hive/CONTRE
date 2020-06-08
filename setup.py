from setuptools import setup

setup(
    name="contre",
    author="Lena Feld",
    author_email="lena.feld@student.kit.edu",
    description="CONTinuum REweighting for Belle and Belle II",
    install_requires=[
        "b2luigi",
        "Sphinx",
        "pandas",
        "root-pandas",
        "scikit-learn",
        "numpy",
        "jupyter"],
    )
