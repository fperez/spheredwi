from __future__ import division
import numpy as np

def spherical_harmonics(theta, phi, N):
	"""Creates vectors of spherical harmonics of length N.
			For theta use gradient_theta and for phi use gradient_phi.
			Initial vector is all 1s.
			At the end all vectors are appended together to create one long vector.
	"""
	
	iso = np.ones((N,1))
	
	Y_neg2 = np.sqrt(30/np.pi)*(1/8)*(np.sin(theta)**2)*(np.e**(-2j*phi))
	Y_neg2 = Y_neg2.reshape(N,1)
	Y_neg2 = np.sqrt(2)*Y_neg2.real
	
	Y_neg1 = np.sqrt(15/(2*np.pi))*(1/2)*(np.sin(theta)*np.cos(theta))*(np.e**(-1j*phi))
	Y_neg1 = Y_neg1.reshape(N,1)
	Y_neg1 = np.sqrt(2)*Y_neg1.real
	
	Y_0 = np.sqrt(5/(4*np.pi))*(1/2)*(3*(np.cos(theta)**2)-1)
	Y_0 = Y_0.reshape(N,1)
		
	Y_1 = np.sqrt(5/(24*np.pi))*-3*(np.sin(theta)*np.cos(theta))*(np.e**(1j*phi))
	Y_1 = Y_1.reshape(N,1)
	Y_1 = np.sqrt(2)*Y_1.imag
	
	Y_2 = np.sqrt(5/(96*np.pi))*3*(np.sin(theta)**2)*(np.e**(2j*phi))
	Y_2 = Y_2.reshape(N,1)
	Y_2 = np.sqrt(2)*(-1)*Y_2.imag

			
	appendment = np.hstack((iso, Y_neg2, Y_neg1, Y_0, Y_1, Y_2))
	
	return appendment
		
