import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import scipy
from scipy import stats
import matplotlib.pyplot as plt

class KFparAlt:
    def __init__(self,  cLin = 10E-3, cMeas = 10E+3):
        ''' Initialize:
        Create object with values for R and Q.
        X[0] with the first measure
        '''
        self.X = np.zeros(2) # X[0] contains the current estimate of the Beam mean X[1] is the estimate of the slope
        self.PX = np.zeros([2, 2]) # Covariance Matrix of X (Calculated by the filter)
        self.Sig = np.zeros(2) # Sig[0] contains the current estimate of Beam std
        self.PS = np.zeros([2, 2]) # Covariance Matrix of Sig (Calculated by the filter)
        self.Q = np.ones([2, 2]) * cLin  # relative confidence in the linear dynamic if increased less noise slower convergence
        self.R = np.array([[cMeas]])  # relative confidence in the measurement if increased faster convergence more noise
        self.F = np.array([[1., 1.], [0, 1.]])  # self dynamic of the system
        # [[link measure to state estimate, link fist order to state estimate],[link state to 1st order, propagate 1st order]]
        self.H = np.array([1., 0])  # link from the state space to the measures space (here the transformation from the measure to X[0] is 1)
        # and we do not measure directly dx/dt but deduce it with F
    #@classmethod
    def EstimateState(self, measure, deltaT):
        # extracting current values from the filter object

        PoldX = self.PX
        PoldSig = self.PS
        oldx = self.X
        oldSig = self.Sig

        F = self.F
        Q = self.Q
        R = self.R
        H = self.H
        F[0, 1] = deltaT # updating F

        # predictions
        xPred = np.dot(F,oldx)  # predicting the state xPred[k,0] = xEst[k-1,0] + (dx/dt)_estimate | xPred[k,1] = (dx/dt)_est
        pPred = np.dot(np.dot(F, PoldX),F.T) + Q  # Covariance matrix of the prediction (the bigger, the less confident we are)

        SigPred = np.dot(F, oldSig)  # Same thing but with standard deviation of the beam current
        SigpPred = np.dot(np.dot(F, PoldSig), F.T) + Q

        Inow = measure

        # updates

        y = Inow - np.dot(H,xPred)  # Calculating the innovation (diff between measure and prediction in the measure space)
        S = np.dot(np.dot(H, pPred),H.T) + R  # Calculating the Covariance of the measure (the bigger the less confident in the measure)

        K = np.dot(np.array([np.dot(PoldX, H.T)]).T, np.linalg.inv(S))  # Setting the Kalman optimal gain
        newX = xPred + np.dot(K, np.atleast_1d(y)).T  # Estimating the state at this instant
        PnewX = np.dot((np.eye(len(pPred)) - np.dot(K, np.array([H]))), pPred)  # Covariance matrix of the state

        # same steps followed for the standard deviation
        y = np.sqrt((Inow - newX[0]) ** 2) - np.dot(H, SigPred)  # Innovation of the standard deviation
        # this is an additional drawer to the Kalman filter and it is rather uncommon to estimate another variable that is
        # statistically dependent there may be better solutions, but this one works
        S = np.dot(np.dot(H, SigpPred), H.T) + R
        K = np.dot(np.array([np.dot(PoldSig, H.T)]).T, np.linalg.inv(S))
        newSig = (SigPred + np.dot(K, np.atleast_1d(y)).T)
        PnewSig = np.dot((np.eye(len(pPred)) - np.dot(K, np.array([H]))), SigpPred)

        # Updating the Kalman filter object
        self.PX = PnewX
        self.X = newX
        self.Sig = newSig
        self.PS = PnewSig







########### Settings
file  = 'd3'
cLin = 10E-2
cMeas = 10E+2
thresh = 8E-4 # threshold on the slope set for the graphic purposes


#loading signal from file

data = np.loadtxt(file)
mes = data[:,1]
time = data[:, 0]

#initializing KF

KF = KFparAlt(cLin = cLin, cMeas = cMeas) # Confidence in lin dynamic, confidence in measurement
# (Adjust the sensitivity of the filter - Confident -> smaller | not confident -> bigger)
# What is important is that the ratio cLin/cMeas is small enough so that the filter can track the dynamic
# of the system and do not lag too much when the direction changes. But this ratio cLin/cMeas has also to
# be big enough so the filter actually filters out the noise and don't just follow the measure where ever it goes.

KF.X[0] = mes[0] # Initializing the filter with the first measure that will stand as the estimate.
# the starting state X will be the measure times the transformation matrix [1, 0] meaning dim1 of X is 1 * measure
# and the dynamic is 0 * measure

# Variables to store the data
esty = np.zeros(len(time))# estimated beam current
estdy = np.zeros(len(time))# estimated first order
ests = np.zeros(len(time))# estimated std

for k in range(len(time)-1):

    deltaT = time[k+1] - time[k] # determining the elapsed time since the last measure
    #KF.F[0, 1] = deltaT # modifying the Transformation matrix F to translate the dynamic of the system

    KF.EstimateState(mes[k+1], deltaT)# Update of the Kalman estimate with the current measure
    # (like in right now, not like in beam current even if in venus case it meas=ns the same)

    # storing the different state estimates
    esty[k+1] = KF.X[0] # Estimate of the tracked variable
    estdy[k+1] = np.abs(KF.X[1]) # Estimate of the slope of the tracked variable
    ests[k+1] = KF.Sig[0] # Estimate of the standard deviation of the tracked variable



plt.figure()
plt.subplot(311)
plt.title('Signal '+file+'; cLin: '+str(cLin)+'; cMeas: '+str(cMeas))
plt.plot(time[1::], mes[1::])
#plt.plot(time, pos)
plt.plot(time[1::], esty[1::])
plt.legend(['measured', 'filtered'])
plt.ylabel('x')
plt.subplot(312)
plt.plot(time[1::], ests[1::])
plt.ylabel(r'$\hat{\sigma}_x$')
plt.subplot(313)
plt.plot(time[1::], estdy[1::])
plt.plot(time[1::], thresh * np.ones(len(time[1::])), '--')
plt.ylabel(r'$|\frac{dx}{dt}|$')
plt.xlabel('time')

plt.show()