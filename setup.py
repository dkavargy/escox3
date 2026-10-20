from setuptools import setup, find_packages

setup(
    name="escox3",
    version="3.0.0",
    packages=find_packages(exclude=("tests", "evaluation")),
    install_requires=[r.split("#")[0].strip() for r in open("requirements.txt") if r.split("#")[0].strip()],
    include_package_data=True,
    package_data={
        "esco_skill_extractor": ["data/*.csv", "static/*", "templates/*"],
        "escox3": ["data/raw/*", "data/reference/*", "templates/*", "static/css/*", "static/js/*", "static/img/*"],
    },
    entry_points={"console_scripts": ["escox3=escox3.__main__:main"]},
    author="Konstantinos Petrakis (ESCOX); ESCOX 3.0 extension",
    description="ESCOX 3.0 - green skills, occupations (ISCO-08) and sectors (NACE Rev. 2.1) from unstructured text",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    classifiers=["Programming Language :: Python :: 3", "License :: OSI Approved :: MIT License"],
)
