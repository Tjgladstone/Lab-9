#Part 1: Measuring the Frequency-Dependence of a Low-Pass RC Filter
import numpy as np
import scipy.optimize as opt
from scipy.interpolate import interp1d import matplotlib
import matplotlib.pyplot as plt
#Data
Frequency = [10, 100, 500, 1000, 5000, 7500, 10000]
Vo = [6.6, 8, 7.6, 7.6, 7.6, 7.6, 7.6]
Vc = [6.6, 6.8, 2.4, 1.2, 0.24, 0.16, 0.12]
phase_data = [3.6, 31.68, 72, 79.2, 86.4, -2.16, -13.68]
# make the arrays
frequency = np.array(Frequency)
Vo = np.array(Vo)
Vc = np.array(Vc)
phase = np.array(phase_data)*3.1415/180
ratio = Vc/Vo
angular_frequency = 2*3.1415*frequency
# Plot(everything)
plt.figure(figsize=(8,6))
plt.loglog(angular_frequency,ratio,'b', linestyle='None', marker='o') plt.ylabel("Log(Vc/Vo)")
plt.xlabel("Log(Angular frequency)")
plt.grid(True, which="both", ls="--")
plt.title("Vc/Vo vs Angular Frequency, loglog plot")
# Simple harmonic motion using acceleration
def given_fit(w, A, tau): # inputs: t:time, A:amplitude, B:
↪offset, w:angular frequency, phi:phase
return A/((1 + ((w*tau)**2))**(0.5)) # output: acceleration
start_pars=[1,1.5*10**(-3)]
#pars, cov = opt.curve_fit(oscillator_model, tvals-start_time, yvals,␣
↪p0=start_pars, betainit)
pars, cov = opt.curve_fit(given_fit, angular_frequency, ratio, p0=start_pars)
A,tau = pars
ypred = given_fit(angular_frequency, A, tau)
# Calculate the errors for A and tau
perr = np.sqrt(np.diag(cov)) A_err, tau_err = perr print(f' tau = {tau}') print(f' A = {A}')
print(f' cov = {cov}')
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
# Function to find the phase shift
def phase_shift_line(w, tau2): return np.arctan(-w * tau2)
#data
Frequency = np.array([10, 100, 500, 1000, 5000, 7500, 10000])
angular_frequency = 2 * np.pi * Frequency
phase_data = np.array([3.6, 31.68, 72, 79.2, 86.4, -2.16, -13.68]) * np.pi /␣
↪180 # Convert degrees to radians
# Theoretical phase shift using an estimated tau from before
theory_phase_shift_line = phase_shift_line(angular_frequency, 0. ↪0009729195045287938)
# fitting the phase data to the phase shift function
phase_pars, phase_cov = opt.curve_fit(phase_shift_line, angular_frequency,␣ ↪-phase_data, p0=[0.9729])
tau = phase_pars[0]
# predict phase shift based on the fitted tau
phase_pred = phase_shift_line(angular_frequency, tau)
#plot experimental data, theoretical line, and fitted line
plt.figure(figsize=(8, 6))
plt.semilogx(angular_frequency, -phase_data, label='Data')
plt.semilogx(angular_frequency, theory_phase_shift_line, 'g-',␣
↪label='Theoretical Line (tau = .0009729 s)') plt.semilogx(angular_frequency, phase_pred, 'r-', label=f'Fitted Line (tau =␣
↪{tau:.5f} s)')
# Label everything all good and such
plt.xlabel('Angular Frequency (rad/s)')
plt.ylabel('Phase Shift (radians)')
plt.title('Phase Shift vs Angular Frequency (Semi-log Plot)') plt.legend(loc='best')
plt.grid(True, which="both", linestyle="--")
plt.show()
 import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
# create a function to fit
def given_fit(w, A, tau):
return A / np.sqrt(1 + (w * tau)**2)
#data
angular_frequency = 2 * np.pi * np.array([10, 100, 500, 1000, 5000, 7500,␣ ↪10000])
ratio = np.array([6.6/6.6, 6.8/8, 2.4/7.6, 1.2/7.6, 0.24/7.6, 0.16/7.6, 0.12/7. ↪6])
#fittin da data
start_pars = [1, 1.5 * 10**(-3)] #an initial guess for A and tau
4
 pars, cov = opt.curve_fit(given_fit, angular_frequency, ratio, p0=start_pars)
A, tau = pars
# Predicted values from the fit
ypred = given_fit(angular_frequency, A, tau)
# find the error for A and tau using the covariance matrix
perr = np.sqrt(np.diag(cov))
A_err, tau_err = perr
# Plot all the gabagoo (data and fit)
plt.figure(figsize=(8, 6))
plt.loglog(angular_frequency, ratio, 'bo', label='Experimental Data') plt.loglog(angular_frequency, ypred, 'r-', label=f'Fit: A = {A:.3f} ± {A_err:.
↪3f}, tau = {tau:.3e} ± {tau_err:.3e} s')
# label it all up nice and goood
plt.xlabel("Angular Frequency (rad/s)")
plt.ylabel("Vc / Vo")
plt.title("Vc/Vo vs Angular Frequency (Log-Log Plot)") plt.grid(True, which="both", linestyle="--") plt.legend(loc='best')
plt.show()
# print the fit parameters along with their errors
print(f"Fit Parameters:")
print(f"A = {A:.3f} ± {A_err:.3f}")
print(f"tau = {tau:.3e} ± {tau_err:.3e} seconds")

# Part 2 Analysis
import numpy as np
import scipy.optimize as opt
from scipy.interpolate import interp1d
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
Frequency = [10,100,500,1000,5000,25000,50000]
Vo = [6.2,7.6,7.6,7.6,8,8,8]
Vc = [0.4,0.56,2.1,4,7.6,8,8]
Phase = [-7.2,-36,-57.6,-50.4,-18,1.8,3.6]
frequency = np.array(Frequency)
Vo = np.array(Vo)
Vc = np.array(Vc)
phase = np.array(Phase)*3.1415/180
ratio = Vc/Vo
angular_frequency = 2*3.1415*frequency
plt.loglog(angular_frequency,ratio,'b',marker='o')
plt.ylabel("Log(VL/Vo)")
plt.xlabel("Log(Angular frequency)")
plt.title("VL/Vo vs Angular Frequency, loglog plot")
plt.plot()
plt.show()
def given_fit(w, A, tau):
    return A * w * tau / np.sqrt(1 + (w * tau)**2)
start_pars = [1, 1e-3]
pars, cov = opt.curve_fit(given_fit, angular_frequency, ratio, p0=start_pars)
A,tau = pars
errors = np.sqrt(np.diag(cov))
error_A, error_tau = errors
print(f"Errors: error_A = {error_A}, error_tau = {error_tau}")
ypred = given_fit(angular_frequency, A, tau)
print(tau)
print(A)
print(cov)
plt.loglog(angular_frequency,ratio,'b',marker='o')
plt.loglog(angular_frequency,ypred,'r',marker='o')
plt.ylabel("Log(VL/Vo)")
plt.xlabel("Log(Angular frequency)")
plt.title("VL/Vo vs Angular Frequency, loglog plot")
plt.plot()
plt.show()
R = 10000
L = 0.1
def phase_shift_line(w, tau2):
    return -np.arctan(-w * tau2) -1.6  # Adjust based on your theoretical model
theory_phase_shift_line = phase_shift_line(angular_frequency, 0.00022)
initial_tau = 0.1  
phase_pars, phase_cov = opt.curve_fit(phase_shift_line, angular_frequency, phase, p0=[initial_tau])
tau = phase_pars
phase_errors = np.sqrt(np.diag(phase_cov))
error_tau_phase = phase_errors[0]
print(f"Error for phase tau = {error_tau_phase}")
print(tau)
phase_pred = phase_shift_line(angular_frequency, tau)
plt.semilogx(angular_frequency,phase)
plt.semilogx(angular_frequency,theory_phase_shift_line)
plt.semilogx(angular_frequency,phase_pred)
plt.xlabel("Angular Frequency (Hz)")
plt.ylabel("Phase Shift (Radians)")
plt.title("Phase Shift vs Angular Frequency")
plt.grid()
plt.plot()
plt.show()
print(tau)
theoretical_tau = tau
def theoretical_phase_shift(w, tau):
    return np.arctan(-w * tau)
theory_phase_shift_line = theoretical_phase_shift(angular_frequency, theoretical_tau)
plt.figure()
plt.semilogx(angular_frequency, phase, 'bo', label='Measured Phase Data')
plt.semilogx(angular_frequency, theory_phase_shift_line, 'r-', label='Theoretical Phase Shift')
plt.xlabel("Angular Frequency (rad/s)")
plt.ylabel("Phase Shift (radians)")
plt.title("Phase Shift vs Angular Frequency")
plt.grid()
plt.show()
