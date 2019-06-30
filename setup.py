from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy
import os
import imp

if os.name =='nt' :
    ext_modules=[
        Extension("detector.cython_utils.nms",
            sources=["detector/cython_utils/nms.pyx"],
            #libraries=["m"] # Unix-like specific
            include_dirs=[numpy.get_include()]
        ),
        Extension("detector.cython_utils.cy_yolo2_findboxes",
            sources=["detector/cython_utils/cy_yolo2_findboxes.pyx"],
            #libraries=["m"] # Unix-like specific
            include_dirs=[numpy.get_include()]
        )
    ]

elif os.name =='posix' :
    ext_modules=[
        Extension("detector.cython_utils.nms",
            sources=["detector/cython_utils/nms.pyx"],
            libraries=["m"], # Unix-like specific
            include_dirs=[numpy.get_include()]
        ),
        Extension("detector.cython_utils.cy_yolo2_findboxes",
            sources=["detector/cython_utils/cy_yolo2_findboxes.pyx"],
            libraries=["m"], # Unix-like specific
            include_dirs=[numpy.get_include()]
        )
    ]

else :
    ext_modules=[
        Extension("detector.cython_utils.nms",
            sources=["detector/cython_utils/nms.pyx"],
            libraries=["m"] # Unix-like specific
        ),
        Extension("detector.cython_utils.cy_yolo2_findboxes",
            sources=["detector/cython_utils/cy_yolo2_findboxes.pyx"],
            libraries=["m"] # Unix-like specific
        )
    ]

setup(
    ext_modules = cythonize(ext_modules)
)
