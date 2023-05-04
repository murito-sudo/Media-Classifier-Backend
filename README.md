# Media-Classifier-Backend

<h1>.Installation guide</h1>
Please see the instructions below to run the program:<br />
<h2>Backend Guide:</h2><br />
a-)Unzip or clone the backend files and place them into a python project folder.<br />
b-)Make sure you have python 3.10 or greater Installed in the system you are accessing the program from.<br />
c-)Load the files from the backend folder into an IDE that can support python 3.10 (We used eclipse for the code development, but you can use any IDE that meets the requirements).<br />
d-) Install the following libraries: Flask, matplotlib, Ipython, requests, io, keras, os, OpenCV, Pillow, NumPy and pandas.<br />
e-)Open the python file called Image Classifier and run the program(If you want to create a new model, delete the model presented in the project file or move it elsewhere if you want to keep it).<br />
<br />
<h2>Frontend Guide</h2><br />
a-)Unzip or clone the  the frontend files here:https://github.com/murito-sudo/Media-Classifier-Client and place them in the desired place.<br />
b-)Open a command prompt with npm installed.<br />
c-) Once on the command prompt with npm installed, use the cd command until you get to the frontend directory.<br />
d-) type npm start(Note: the frontend won’t retrieve images nor predictions until the backend server is activated. You can activate it by running the program in the IDE, we’ll explain below another method to initialize the backend server).<br />
<br />
<h2>Backend Activation Way #2:</h2><br />
a-)Open the command prompt.<br />
b-)Once on the command prompt, navigate through the python project folder.<br />
c-) Once on the python project folder, type this in  the command prompt: python -m venv /path/to/python/project/folder<br />
d-) Inside the venv folder, install these packages: Flask, matplotlib, Ipython, requests, io, keras, os, OpenCV, Pillow, NumPy, pandas, and requests.<br />
c-)Once the virtual environment is created, type this in the command prompt: env\Scripts\activate<br />
d-)You’re going to see an (env) on the command prompt, once that happens, type this in the command prompt: python “Image Classifier.py”.<br />
