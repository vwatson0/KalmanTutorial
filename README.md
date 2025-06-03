# KalmanTutorial

Created June 3, 2025

Contains a tutorial, codes, functions, examples and detailed pdf for the use of a Kalman filter to estimate the beam current mean standard deviation and slope online.

The pdf contains detail about Kalman filtering, how it is adapted to the specific needs of VENUS and show examples to help the user setting up the parameters of the filter.

Code storing the variable and Parameters of the Kalman filter:

![alt text](https://github.com/vwatson0/KalmanTutorial/blob/main/ObjectKFpar.png)

Code of the function updating de Kalman filter:

![alt text](https://github.com/vwatson0/KalmanTutorial/blob/main/FunctionEstimateState.png)

Settings for KFtutorial.py:

![alt text](https://github.com/vwatson0/KalmanTutorial/blob/main/SettingsTutorial.png)

To settup cLin and cMeas see KFtutorial.pdf, select file 'd1' 'd2' or 'd3' or replace by a text file with a list with [time space_separator measure \n]
Threshold is just the value of the dashed line plot with the slope estimate 

Settings for KFtutorialAtan.py

![alt text](https://github.com/vwatson0/KalmanTutorial/blob/main/SettingsTutorialAtan.png)

Kalman filter settings work the same way than KFtutorial.py
The user can change the signal generated between line 95 and 103

##############################################################################################################

AlternateKFtutorial.py :

Alternative to Kalman filter using a method instead of a function.
Result are the same, the code is just more compact.
The Initialization of the object KFparAlt is the same as the object KFpar

name  = KFparAlt(cLin, cMeas)

Initialization with the first measure is also the same:

name.X[0] = measure@T=0

Changes regard the update of the estimate.
NO NEED TO UPDATE F BEFORE THE OBJECT ANYMORE

Just call the method:

name.EstimateState(Newmeasure, deltaT) 

and it will update the object

Results are accessible the same way:

Current : name.X[0]; slope: name.X[1]; Stdev name.Sig[0]

