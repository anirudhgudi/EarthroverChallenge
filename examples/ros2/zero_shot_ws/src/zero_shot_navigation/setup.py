import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'zero_shot_navigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mango',
    maintainer_email='anirudhgudi66@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'zero_shot_perception = zero_shot_navigation.zero_shot_perception:main',
            'zero_shot_planner = zero_shot_navigation.zero_shot_planner:main'
        ],
    },
)
