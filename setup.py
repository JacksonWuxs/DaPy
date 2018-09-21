from setuptools import setup, find_packages
from DaPy import __version__
 
setup(
    name='DaPy',
    version=__version__,
    description='Enjoy your tour in data minning',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
    ],
    author='Xuansheng Wu',
    author_email='wuxsmail@163.com',
    maintainer='Xuansheng Wu',
    maintainer_email='wuxsmail@163.com',
    platforms=['all'],
    url='https://github.com/JacksonWoo/DaPy',
    license='GPL v3',
    packages=['datasets', 'machine_learn', 'statis'],
    package_dir={'datasets': 'DaPy/datasets',
                 'machine_learn': 'DaPy/machine_learn',
                 'statis': 'DaPy/statis'},
    package_data={'datasets': ['adult/*.*', 'example/*.*', 'iris/*.*', 'wine/*.*']},
    zip_safe=True,
    install_requires=[
        'savReaderWriter>=3.4.1',
        'xlrd>=1.1.0',
        'xlwt>=1.3.0',
    ]

)
