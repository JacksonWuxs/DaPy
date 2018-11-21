from setuptools import setup, find_packages
pkg = find_packages()

setup(
    name='DaPy',
    version='1.5.4.4',
    description='Enjoy your tour in data mining',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
    ],
    author='Xuansheng Wu',
    author_email='wuxsmail@163.com',
    maintainer='Xuansheng Wu',
    maintainer_email='wuxsmail@163.com',
    platforms=['all'],
    url='http://dapy.kitgram.cn',
    license='GPL v3',
    packages=pkg,
    package_dir={'DaPy.datasets': 'DaPy/datasets'},
    package_data={'DaPy.datasets': ['adult/*.*', 'example/*.*', 'iris/*.*', 'wine/*.*']},
    zip_safe=True,
    install_requires=[
        'xlrd>=1.1.0',     # Used for function DataSet.read()
        'xlwt>=1.3.0',     # Used for function DataSet.save()
        'ciso8601>=2.1.1', # Used for function: str2value()
    ]

)
