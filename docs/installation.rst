Installation
============

This document outlines various methods for installing `pyrichlet`.

Installing the Latest Version
-----------------------------

To install the latest version of a package from the Python Package Index
(PyPI), use the following command:

::

 pip install pyrichlet

Installing a Specific Version
-----------------------------

To install a specific version of a package, use the following command:

::

 pip install pyrichlet==<version_number>

Replace <version_number> with the desired version number, for example to install
version `0.0.9`:

::

 pip install pyrichlet==0.0.9


Installing from the Repository
------------------------------

There are two ways to install `pyrichlet` repository:

Using the URL
^^^^^^^^^^^^^

::

 pip install git+https://github.com/cabo40/pyrichlet@<branch>

Replace <branch> with the specific branch you want to install
from (optional, defaults to the default branch `main`).
For example,

::

 pip install git+https://github.com/cabo40/pyrichlet@main


Using a local clone
^^^^^^^^^^^^^^^^^^^


Clone the repository to your local machine.

::

 git clone https://github.com/cabo40/pyrichlet.git


Navigate to the cloned repository directory.

::

 cd pytichlet

Install as a local package:

::

 pip install .

Uninstalling
------------

To uninstall `pyrichlet` regarding of the installation method you need to do:

::

 pip uninstall pyrichlet

Afterwards you can remove any directory created during the installation, for
example if it was installed as a local package.