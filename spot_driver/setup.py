from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['spot_driver', 'spot_driver.arm', 'spot_driver.arm.arm_utilities'],
    scripts=['scripts/spot_ros'],
    package_dir={'': 'src'}
)

setup(**d)
