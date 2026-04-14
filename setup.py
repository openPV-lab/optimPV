import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="optimpv", 
    version="v1.05",
    author="Vincent M. Le Corre",
    author_email="",
    description="optimPV: Optimization & Modeling tools for PV research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='GPLv3',
    url="https://github.com/openPV-lab/optimPV",
    download_url="https://github.com/openPV-lab/optimPV/v1.05.tar.gz",
    packages=setuptools.find_packages(),
    readme = "README.md",
    keywords=['Bayesian optimization', 'Evolutionary optimization','parameter extraction', 'experimental design', 'high throughput', 'solar cells'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    python_requires='>=3.12',
    install_requires = [
        'numpy>=1.2',
        'pandas>=1.4',
        'matplotlib>=3.5',
        'jupyterlab>=3.4',
        'seaborn>=0.11',
        'scipy>=1.0',
        'gitpython>=3.1',
        'openpyxl>=3.0',
        'pyodbc>=4.0',
        'scikit-optimize>=0.9',
        'pySIMsalabim>=1.3',
        'tk',
        'torch>=2.0',
        'torchvision>=0.15',
        'torchaudio>=2.0',
        'emcee>=3.1',
        'pymoo>=0.6',
        'ax-platform>=1.2.4',
        'arviz>=0.15.1',
        'pydantic'
    ],
    extras_require = {
        'dev': [
            'pytest',
            'twine',
        ],
    },
    include_package_data=True,
        
)