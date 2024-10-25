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
