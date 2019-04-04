This program requires the opencv 2 and scikit-image libraries.
The easiest way to get both of them installed correctly is to install the
32-bit Anaconda stack for python: https://www.continuum.io/Downloads
and Anaconda in as Pycharm's Project Interpreter. Anaconda will already
include scikit-image. 

To get opencv2 installed, run command prompt as an administrator.
run the commands:
anaconda -h
conda install -c menpo opencv=2.4.11

Further information can be found at:
https://docs.continuum.io/anaconda-cloud/using#Installing

Once all the libraries have been installed correctly, you can run the
python script to test it. Change the variable 'name' to the full name of
any JPEG image to select it. Image files must be located in the same 
folder as the FaceLocalization project.

Example: name='exercise-1.jpg'