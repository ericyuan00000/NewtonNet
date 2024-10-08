import setuptools
from os import path
import newtonnet

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md')) as f:
    long_description = f.read()

if __name__ == "__main__":
    setuptools.setup(
        name=newtonnet.__name__,
        version=newtonnet.__version__,
        author='Teresa Head-Gordon',
        author_email='thg@berkeley.edu',
        project_urls={
            'Source': 'https://github.com/THGLab/NewtonNet',
        },
        description=
        "A Newtonian message passing network for deep learning of interatomic potentials and forces",
        long_description=long_description,
        long_description_content_type="text/markdown",
        scripts=['scripts/newtonnet_train.py'],
        keywords=[
            'Machine Learning', 'Data Mining', 'Quantum Chemistry',
            'Molecular Dynamics'
        ],
        license='MIT',
        packages=setuptools.find_packages(),
        install_requires=[
            'numpy<2.0',
            'scipy',
            'scikit-learn',
            'pandas',
            'ase',
            'tqdm',
            'pyyaml',
            'wandb',
            'torch',
            'torch_geometric',
            # 'torch_scatter',
            # 'torch_sparse',
            # 'torch_cluster',
            # 'torch_spline_conv',
        ],
        include_package_data=True,
        classifiers=[
            'Development Status :: 4 - Beta',
            'Natural Language :: English',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3',
        ],
        zip_safe=False,
    )
