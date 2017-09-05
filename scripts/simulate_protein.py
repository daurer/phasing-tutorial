#!/usr/bin/env python
import h5py
import condor
import copy
import numpy as np
import condor.utils.linalg as linalg
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

# Detector (pnCCD)
pixelsize = 4*75e-6
nx,ny = (1024//4,1024//4)
detector_distance = 300e-3

# Source
photon_energy = 3500 # [eV]
ph = condor.utils.photon.Photon(energy_eV=photon_energy)
wavelength = ph.get_wavelength()
fluence = 1e15 #[ph/um2]
focus_diameter = 0.2e-6
pulse_energy = fluence * ph.get_energy() * (np.pi*((1e6*focus_diameter/2.)**2)) # [J]

# Sample
pdb_id = '1FFK'
sample_size = 18e-9

# Simulation
src = condor.Source(wavelength=wavelength, pulse_energy=pulse_energy, focus_diameter=focus_diameter)
det = condor.Detector(distance=detector_distance, pixel_size=pixelsize, nx=nx, ny=ny, noise="poisson")
par = condor.ParticleAtoms(pdb_id=pdb_id,rotation_formalism="random")
E = condor.Experiment(source=src, particles={"particle_atoms":par}, detector=det)

# Run multiple times and Save to file
N = 100
W = condor.utils.cxiwriter.CXIWriter("../data/single_protein_diffraction_patterns.h5")
for i in range(N):
    o = E.propagate()
    W.write(o)
W.close()

# Save interpolated 3D diffraction volume
# =======================================
points = []
values = []
N = 100
for i in range(N):
    res = E.propagate()
    img = res["entry_1"]["data_1"]["data_fourier"]
    qmap = E.get_qmap_from_cache()
    c = 2*np.pi * D.pixel_size / (S.photon.get_wavelength() * D.distance)
    points.append(qmap.reshape((qmap.shape[0]*qmap.shape[1], 3)) / c)
    values.append(img.flatten())
points = np.array(points)
points = points.reshape((points.shape[0]*points.shape[1], 3))
values = np.array(values).flatten()

# Interpolation
grid_x, grid_y, grid_z = np.mgrid[0:(nx-1):nx*1j, 0:(ny-1):ny*1j, 0:(nz-1):nz*1j]
grid_x = (grid_x-(nx-1)/2.)
grid_y = (grid_y-(ny-1)/2.)
grid_z = (grid_z-(nz-1)/2.)
grid = (grid_x, grid_y, grid_z)

# Complex valued 3D diffraction space
img_3d = scipy.interpolate.griddata(points, values, grid, method='linear')
intensities_3d = abs(img_3d)**2
phases_3d = np.angle(img_3d)

# Real space object
tmp = img_3d.copy()
tmp[np.isnan(tmp)] = 0.
tmp = np.fft.fftshift(tmp)
rs_3d = np.fft.fftshift(np.fft.ifftn(tmp))

# Save to file
with h5py.File('../data/single_protein_diffraction_volume.h5') as f:
    f['fourier_complex'] = img_3d
    f['fourier_intensities'] = intensities_3d
    f['fourier_phases'] = phases_3d
    f['realspace'] = rs_3d

# Save full 3D fourier volume
# ===========================
W = condor.utils.cxiwriter.CXIWriter("../data/single_protein_fourier_volume.h5")
W.write(E.propagate3d)
W.close()
