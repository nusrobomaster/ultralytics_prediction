from setuptools import find_packages, setup

package_name = 'gimbal'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nicholas_tyy',
    maintainer_email='nicholastanyunyu@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dummy_gimbal_orientation_publisher = gimbal.dummy_gimbal_orientation_publisher:main',
            'yolov8_publisher = gimbal.dummy_yolov8_publisher:main',
            'yolov8_subscriber = gimbal.yolov8_subscriber:main',
            'main = gimbal.main:main'
        ],
    },
)
