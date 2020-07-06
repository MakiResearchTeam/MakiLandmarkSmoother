# Copyright (C) 2020  Igor Kilbas, Danil Gribanov
#
# This file is part of MakiLandmarkSmoother.
#
# MakiLandmarkSmoother is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MakiLandmarkSmoother is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <https://www.gnu.org/licenses/>.

from setuptools import setup
import setuptools

setup(
    name='MakiLandmarkSmoother',
    packages=setuptools.find_packages(),
    version='1.0.0',
    description='This repo contains of tool that allow you to avoid "jitter" of landmarks on the video.',
    long_description='...',
    author='Kilbas Igor, Gribanov Danil',
    author_email='whitemarsstudios@gmail.com',
    url='https://github.com/MakiResearchTeam/MakiLandmarkSmoother',
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ], install_requires=[]
)