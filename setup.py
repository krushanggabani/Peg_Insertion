
from logging import root
import os
import sys
from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install
import subprocess


if __name__ == '__main__':
    setup(
        name='Roboforce Peg Insertion',
        version='1.0',
        description='Pybullet SIMULATOR',
        author='Krushang Gabani',
        keywords='simulation, deformable bodies, soft robotics, Human Robot Interaction, reinforcement learning',
        packages=[    ],
        python_requires = '>=3.7',
        install_requires=[
            "cuda-python",
            "gym",
            "imageio",
            "imageio-ffmpeg",
            "matplotlib",
            "numpy==1.26.3",
            "opencv-python",
            "open3d",
            "pandas",
            "scipy",
            "taichi",
            "torch",
            "torchvision",
            "pyyaml",
            "yacs",
            "tensorboard",
            "pybullet",
            "pygmsh"
        ]
    )
    



