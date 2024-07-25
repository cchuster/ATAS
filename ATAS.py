#%% Import the necessary libraries
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from oct2py import Oct2Py
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from numpy.fft import rfft
#%% Generate the static absorption plot of the sample with respect to pixels
path = '/home/cchuster/Summer 2024/Leone-Group/CrossTe/'
file = 'static_Te_12024_02_05_11_56_53.hdf5'
f = h5py.File(str(path)+file, 'r')
pixel = np.arange(0, 1340, 1)
membrane = f['data/res0/membrane'][:199, :]
sample = f['data/res0/sample'][:199, :]
bkg = np.average(f['data/bg0/off'][:, :], axis=0)

Static_2024_06_27 = np.average(-(np.log(sample/(membrane))), axis=0)
plt.plot(energy,Static_2024_06_27, color='darkred')
plt.fill_between(energy,Static_2024_06_27,0,color='darkred',alpha=0.3)

plt.xlabel("Energy (eV)")
plt.ylabel("Absorption (OD)")
plt.xlim(38, 50)
plt.ylim(0,3)
plt.title("Elemental Tellurium")
#%% Extract absorption data
# Call the edge reference Matlab function
def edge_reference(specOn, specOff, Edge):
    oc = Oct2Py()
    dOD_referenced = oc.edgeReferenceTransient(specOn, specOff, Edge)
    return dOD_referenced

path = '/home/cchuster/Summer 2024/Leone-Group/CrossTe/'
name = 'scan_Te_6_27uJ_round_3scan'

# Store the background and pump-probe data for all cycles
data = []
for file in os.listdir(path):
    if file.startswith(name):
        f = (h5py.File(str(path)+file, 'r'))
        data.append(f)

# Background measurement taken only once, but stored in each pump-probe cycle for good practice
bkg_on = np.average((data)[1]['data/bg0/on'][:, :], axis=0)
bkg_off = np.average((data)[1]['data/bg0/off'][:, :], axis=0)

pumpon = []
pumpoff = []

# Append the data from each of the 1340 pixels at each time delay for each pump-probe cycle 
for i in range(len(data)-1):
    pumpon.append(data[i]['data/res0/on'][:, :])
    pumpoff.append(data[i]['data/res0/off'][:, :])

# Subtract the on/off background noise for each pixel at each time delay for each pump-probe cycle
pumpon = np.array(pumpon)[:, :, :]-bkg_on
pumpoff = np.array(pumpoff)[:, :, :]-bkg_off

print("Dimensions of pumpon:", pumpon.shape)
print("Dimensions of pumpoff:", pumpoff.shape)
#%% Clean copolarized absorption data
# Find the average change in absorption for each of the time delays for each of the pixels across all pump-probe cycles
AbsorptionCo = np.average(-(np.log(pumpon/pumpoff)), axis=0)
AbsorptionCo[np.isnan(AbsorptionCo)] = 0
print("Dimensions of AbsorptionCo:", AbsorptionCo.shape)

# Absorption becomes an edge referenced 2D array, its first dimension corresponding to time delay and second to pixels
pixel = np.arange(0, 1340, 1)
AbsorptionCo = edge_reference(pumpon, pumpoff, ((pixel<200)&(pixel>0))|((pixel<1000)&(pixel>800)))
#%% Clean cross polarized absorption data
# Find the average change in absorption for each of the time delays for each of the pixels across all pump-probe cycles
AbsorptionCross = np.average(-(np.log(pumpon/pumpoff)), axis=0)
AbsorptionCross[np.isnan(AbsorptionCross)] = 0
print("Dimensions of AbsorptionCross:", AbsorptionCross.shape)

# Absorption becomes an edge referenced 2D array, its first dimension corresponding to time delay and second to pixels
pixel = np.arange(0, 1340, 1)
AbsorptionCross = edge_reference(pumpon, pumpoff, ((pixel<200)&(pixel>0))|((pixel<1000)&(pixel>800)))
#%% Generate the transient absorption color plot of the sample with respect to pixels
fig, ax1 = plt.subplots()
plt.xlim(300,600)
# Plot a scaled absorption color map with the lower limit of the color map equal to -2.5 and the upper limit at 2.5
z = ax1.pcolor(pixel, np.arange(AbsorptionCo.shape[0]), AbsorptionCo[:, :]*1000, cmap="jet", vmin=-100, vmax=100, shading='auto')
bar = fig.colorbar(z)
bar.set_label(r"$\Delta$A (mOD)")
ax2 = ax1.twinx()
ax2.set_yticks([])
ax1.set_title("Co-Polarized")
ax1.set_ylabel("Pump-Probe Delay (fs)")
ax1.set_xlabel("Pixels")
# %% Perform Singular Value Decomposition
def SVD_component(b, component):
    b[np.isnan(b)] = 0
    # Create the rotation, scale, rotation matrices
    U, s, Vt = np.linalg.svd(b, full_matrices=False)
    # Transpose the V matrix
    V = Vt.T
    # The "prevelance" value of the selected component
    print(s[component])
    # Reconstruct the original transient signal
    signal = np.outer(U[:, component], V[:, component].T)*s[component]
    # The temporal and spectral signals (weighted by the prevelance of the singal)
    temporal_signal = U[:, component]*s[component]
    spectral_signal = Vt[component, :] * s[component]
    return signal, temporal_signal, spectral_signal

plt.xlabel("Time Delay")
plt.ylabel("Amplitude")
plt.xlim(-100,2000)

qCross = AbsorptionCross[:,:]*1000
qCo = AbsorptionCo[:,:]*1000

timeCross = (np.arange(-20, 450, 2))*6.6
timeCo = (np.arange(-187, 168, 4)+136)*6.6

sigCross, tCross, sCross = SVD_component(
    qCross[:,350:550], 0)

sigCo, tCo, sCo = SVD_component(
    qCo[:,350:550], 0)

# Normalize and plot the cross-polarized and co-linear temporal signals
plt.plot(timeCross, tCross/np.max(tCross), marker='o', markersize=6, linestyle='--',label='Cross-Polarized')
plt.plot(timeCo, -tCo/np.max(-tCo), marker='o', markersize=6, linestyle='--',label='Co-Polarized')
plt.legend()

f = interp1d(timeCo, tCo, kind='cubic')
interpolated_tCo = f(timeCross[:150])
subtracted_signal = tCross[:150]/np.max(tCross[:150]) + interpolated_tCo/np.max(-interpolated_tCo)
plt.figure()
plt.plot(timeCross[:150], -subtracted_signal/np.max(-subtracted_signal), marker='o', linestyle='-', label='Subtracted Signal', color='forestgreen')
plt.xlabel("Time Delay")
plt.ylabel("Amplitude")
plt.legend()
#%% Generate the transient absorption color plot of the sample with respect to energy
# Calibrate the conversion between the pixels excited after grating and the energy level
# Pixel position and energy are related non-linearly, so approximate to second order
def func(x, a, b, c):
    return a+x*b+c*x**2

# Retrieved via transient Neon gas scan
selected_lines = np.array([617, 671, 1021, 1091])
selected_energies = np.array([45.54, 47.12, 60, 63.6])

# Return the values of each coefficient in the second order polynomial function and the estimated covariance
popt, pcov = curve_fit(func, selected_lines, selected_energies, bounds=(0, [50, 50, 0.1]))
# Set energy equal to the second order polynomial function
energy = func(pixel, *popt)

fig, ax1 = plt.subplots()
plt.xlim(38,45)
plt.ylim(-50, np.max(timeCo))
# Plot a scaled absorption color map with the lower limit of the color map equal to -2.5 and the upper limit at 2.5
z = ax1.pcolor(energy, timeCross, AbsorptionCross[:, :]*1000, cmap="jet", vmin=-50, vmax=50, shading='auto')
bar = fig.colorbar(z)
bar.set_label(r"$\Delta$A (mOD)")
ax2 = ax1.twinx()
ax2.set_yticks([])
ax1.set_title("Cross-Polarized")
ax1.set_ylabel("Pump-Probe Delay (fs)")
ax1.set_xlabel("Energy (eV)")

# %% Find the real component of the Fast Fourier Transform of the temporal SVD component
tCross_fft = rfft(np.pad(tCross, (0, len(tCross)), 'constant'))
tCo_fft = rfft(np.pad(tCo, (0, len(tCo)), 'constant'))
freqCross = np.fft.rfftfreq(len(tCross)*2, 0.002*3.335668*2)
freqCo = np.fft.rfftfreq(len(tCo)*2, 0.004*3.335668*2)
# Plot the Fourier Transform
plt.figure()
plt.plot(freqCross, np.abs(tCross_fft), label='FT of Cross-Polarized SVD')
plt.plot(freqCo, np.abs(tCo_fft), label='FT of Co-Polarized SVD')
plt.xlim(0,6)
plt.ylim(0,10000)
plt.xlabel('Frequency (THz)')
plt.ylabel('Amplitude')
plt.title('Fourier Transform of Temporal SVD')
plt.legend()
plt.show()
# %% Alternating waveplate scans
# Call the edge reference Matlab function
def edge_reference(specOn, specOff, Edge):
    oc = Oct2Py()
    dOD_referenced = oc.edgeReferenceTransient(specOn, specOff, Edge)
    return dOD_referenced

path = '/home/cchuster/Summer 2024/Leone-Group/AltTe/'
name = 'scan_Te_waveplate_altscan'

# Store the background and pump-probe data for all cycles
data = []
for file in os.listdir(path):
    if file.startswith(name):
        f = (h5py.File(str(path)+file, 'r'))
        data.append(f)

# Background measurement taken only once, but stored in each pump-probe cycle for good practice
bkg_on = np.average((data)[1]['data/bg0/on'][:, :], axis=0)
bkg_off = np.average((data)[1]['data/bg0/off'][:, :], axis=0)

pumpon = []
pumpoff = []

# Append the data from each of the 1340 pixels at each time delay for each pump-probe cycle 
# Set zero for copolarized and one for cross polarized, then go back and run the above and then run the SVD below
for i in range(len(data)-1):
    pumpon.append(data[i]['data/res0/on'][:,1,:])
    pumpoff.append(data[i]['data/res0/off'][:,1,:])

# Subtract the on/off background noise for each pixel at each time delay for each pump-probe cycle
pumpon = np.array(pumpon)[:, :, :]-bkg_on
pumpoff = np.array(pumpoff)[:, :, :]-bkg_off

print("Dimensions of pumpon:", pumpon.shape)
print("Dimensions of pumpoff:", pumpoff.shape)
# %% Perform Singular Value Decomposition on alternating waveplate scan
def SVD_component(b, component):
    b[np.isnan(b)] = 0
    # Create the rotation, scale, rotation matrices
    U, s, Vt = np.linalg.svd(b, full_matrices=False)
    # Transpose the V matrix
    V = Vt.T
    # The "prevelance" value of the selected component
    print(s[component])
    # Reconstruct the original transient signal
    signal = np.outer(U[:, component], V[:, component].T)*s[component]
    # The temporal and spectral signals (weighted by the prevelance of the singal)
    temporal_signal = U[:, component]*s[component]
    spectral_signal = Vt[component, :] * s[component]
    return signal, temporal_signal, spectral_signal

qCross = AbsorptionCross[:,:]*1000
qCo = AbsorptionCo[:,:]*1000

timeCross = (np.arange(-44, 450, 4))*6.6
timeCo = (np.arange(-44, 450, 4))*6.6

sigCross, tCross, sCross = SVD_component(
    qCross[:,350:550], 0)

sigCo, tCo, sCo = SVD_component(
    qCo[:,350:550], 0)

plt.plot(timeCross, tCross/np.max(tCross), marker='o', markersize=6, linestyle='--',label='Cross-Polarized')
plt.plot(timeCo, tCo/np.max(tCo), marker='o', markersize=6, linestyle='--',label='Co-Polarized')
plt.legend()
plt.xlabel("Time Delay")
plt.ylabel("Amplitude")
plt.legend()
plt.show()
# %%
