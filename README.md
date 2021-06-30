# Motor-Imagery-Classification-on-Brain-EEG-Signal
This project aim is to classify the motor imagery signals extracted from the brain (using an Electro Encephalogram) using ML 

The data set for the project is obtained by requeisting bci organisation 
http://www.bbci.de/competition/iv/

The obtained dataset can be found from
https://drive.google.com/drive/folders/1iMc6bimGLAnnElXuvmHI9noCC7vMNiC9?usp=sharing

The kind of dataset used here is 2 class motor imagery, which deals with right, left hands and legs(based on the chosen set)

The next step is to convert read those into python used suitable modules and then converting the time domain signals into frequency domain using PSD techinque(Power Spectral Density)

The next part is to see the difference between various frequencies of signsla(delta, theta, alpha, beta, gamma) and how they varies for each class. 

From various resources from net and some articles, We came to a conclusion that mostly the sigals which are in the range of alpha - low beta (8hz - 15hz) contribute for
mind - body coordination, integration and performing tasks.

So now, to filter these signals, we used scipy.signals library and plotted the filtered signals and along with the difference between psd values of both classes in the given 
frequency range



