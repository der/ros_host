from setuptools import find_packages, setup

package_name = 'audio_base'

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
    maintainer='der',
    maintainer_email='dave@epimorphics.com',
    description='Basic audio processing and testing for Marvin',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'capture_node = audio_base.capture_node:main',
            'player_node = audio_base.player_node:main',
            'asr_node = audio_base.asr_node:main',
            'tts_node = audio_base.tts_node:main',
        ],
    },
)
