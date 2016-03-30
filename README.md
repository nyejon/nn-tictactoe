# nn-tictactoe
Testing NNs to learn the simple game of TicTacToe


# Installation

Install Python 2.7 64-bit. We are running `Python 2.7.9 (default, Dec 10 2014, 12:28:03) [MSC v.1500 64 bit (AMD64)] on win32`

Install Visual C++ compiler for Python from: <https://www.microsoft.com/en-us/download/details.aspx?id=44266> **NOTE:** run the following command to install the compiler: `msiexec /i VCForPython27.msi ALLUSERS=1`

Download stdint.h from http://msinttypes.googlecode.com/svn/trunk/stdint.h and then copy it into `C:\Program Files (x86)\Common Files\Microsoft\Visual C++ for Python\9.0\VC\include`

Download and install TDM GCC from http://tdm-gcc.tdragon.net/ (take 64-bit bundle). We are using GCC 5.1.0.

Open a command window and run the following:

* `"C:\Program Files (x86)\Common Files\Microsoft\Visual C++ for Python\9.0\vcvarsall.bat" amd64`
* `path C:\TDM-GCC-64\bin;%PATH%`
* `path C:\TDM-GCC-64\x86_64-w64-mingw32\bin;%PATH%`

Install wheels from http://www.lfd.uci.edu/~gohlke/pythonlibs/ for:

* numpy (1.11.0)
* scikit-learn (0.17.1)
* scipy (0.17.0)
* Theano (0.8.1)

To get Theano to work, you have to recompile libpython.a - follow instructions on:

* http://rosinality.ncity.net/doku.php?id=python:installing_theano

**NOTE:** *gendef* and *dlltool* are available in TDM GCC by adding `C:\TDM-GCC-64\x86_64-w64-mingw32\bin` to the PATH.

# Make sure everything is working

If you do the above you should be able to open python and `import theano` without getting any errors or warnings.