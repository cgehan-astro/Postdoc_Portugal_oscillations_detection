### Create the appropriate environment with anaconda: conda env create -f environment.yml
### python=3.9.25, ipython=8.15


### This program aims at detecting oscillations in stars due to internal standing waves, by detecting and measuring the frequency spacing between consecutive oscillation modes in the frequency power spectrum, i.e the large separation Dnu, and its uncertainty, a fundamental parameter needed for stellar physics 

import numpy as np
from scipy.fftpack import fft, ifft, fftfreq
import pylab as plt
import glob
import time



### Routine windowing the power spectrum with a Hanning filter		

def function_windowing_spectrum_Hanning(NH, nu_max, dnuH, freq, P):
	Hanning = np.hanning(NH)				# defining the Hanning filter
	nuH = np.linspace(nu_max - dnuH, nu_max + dnuH, NH)	# defining the frequency range selected by the filter
	if np.min(nuH) < np.min(freq):
		nuH = np.linspace(np.min(freq), nu_max + dnuH, NH)
	if np.max(nuH) > np.max(freq):
		nuH = np.linspace(nu_max - dnuH, np.max(freq), NH)
	freq_corr = freq[np.where(freq >= np.min(nuH))[0]]		# selecting frequencies and power spectral densities in the range of frequencies covered by the Hanning filter
	PSD_corr = P[np.where(freq >= np.min(nuH))[0]]
	freq_corr = freq_corr[np.where(freq_corr <= np.max(nuH))[0]]
	PSD_corr = PSD_corr[np.where(freq_corr <= np.max(nuH))[0]]
	Hanning_modes = np.interp(freq_corr, nuH, Hanning)		# interpolating the Hanning filter at observed frequencies
	spectrum = PSD_corr * Hanning_modes				# windowing the power spectrum with the Hanning filter
	return freq_corr, spectrum



### Routine computing the autocorrelation of the spectrum: taking the Fourier spectrum of the spectrum

def function_autocorrelation(spectrum, zero_padding_factor, freq_corr):
	N_FFT = zero_padding_factor * N_spectrum				# defining the number of zeros to add to zero-pad the spectrum
	spectrum = np.pad(spectrum, (0, N_FFT), mode='constant')			# zero-padding the spectrum to increase the future resolution in the time domain
	FFT = ifft(spectrum)							# taking the inverse Fourier transform of the zero-padded spectrum
	dnu = (np.max(freq_corr) - np.min(freq_corr)) / N_spectrum * 10.**(-6)	# frequency resolution of the zero-padded spectrum, in Hz
	t = fftfreq(spectrum.size)	 					# dimensionless time
	tau_plus = np.where(t >= 0)[0]
	FFT = FFT[tau_plus]							# selecting only the part of the Fourier spectrum associated to positive time
	tau = np.arange(spectrum.size/2) / (spectrum.size * dnu) / 3600	 	# constructing the time, in hours
	return tau, FFT, dnu



### Routine computing the Envelope Autocorrelation Function and taking the maximum value to estimate the frequency regular spacing, i.e. the large separation Dnu

def function_EACF(tau_min, tau_max, spectrum, zero_padding_factor, freq_corr, N_spectrum, N_oversampling):
	tau, FFT, dnu = function_autocorrelation(spectrum, zero_padding_factor, freq_corr)	# autocorrelation of the windowed spectrum; tau is the time axis, in hours
	A_star = np.abs(FFT**2) / np.abs(FFT[0]**2)							# normalizing the autocorrelation
	sigma_H = 3./(2 * (N_spectrum/N_oversampling - 1))						# estimating the noise level of the autocorrelation
	EACF_star = A_star / sigma_H									# computing the Envelope Autocorrelation Function: normalized autocorrelation over
													# the noise level
			
													
	## Selecting a more restrictive period range to find the maximum of the Envelope Autocorrelation Function

	Dnu_max = 2. * 10.**6 /(tau_min * 3600)
	if Dnu_max > Dnu_max_to_test:					# limiting the upper Dnu value to be searched for
		tau_min = 2. * 10.**6 /(Dnu_max_to_test * 3600)
	crit_tau_min = np.where(tau >= tau_min)
	tau_EACF = tau[crit_tau_min]
	EACF_star = EACF_star[crit_tau_min]
	crit_tau_max = np.where(tau_EACF <= tau_max)
	tau_EACF = tau_EACF[crit_tau_max]
	EACF_star = EACF_star[crit_tau_max]


	## Taking the maximum of the Envelope Autocorrelation Function and estimating Dnu

	Dnu_crit = np.where(EACF_star == np.max(EACF_star))[0]
	tau_Dnu = tau_EACF[Dnu_crit]
	EACF_Dnu = EACF_star[Dnu_crit]
	Dnu_found_ini = 2. * 10.**6 /(tau_Dnu * 3600)
	return Dnu_found_ini, EACF_Dnu, tau_Dnu, tau_EACF, EACF_star, dnu



### Routine measuring Dnu, its corresponding Envelope Autocorrelation Function values, the center of the Hanning filter used to window the spectrum: where the Envelope Autocorrelation Function is maximum

def function_Dnu_measurement(all_Dnu_final, all_EACF_final, all_Hanning_centers_final):	 
	max_EACF_final = np.max(all_EACF_final)
	crit_max_EACF = np.where(all_EACF_final == max_EACF_final)[0]
	Dnu_final = all_Dnu_final[crit_max_EACF]			
	EACF_final = all_EACF_final[crit_max_EACF]			
	crit_high_EACF = crit_max_EACF	
	Hanning_centers_final = all_Hanning_centers_final[crit_max_EACF]				
	return Dnu_final, EACF_final, Hanning_centers_final



### Definition of needed parameters

N_oversampling = 1.			# oversampling factor of the spectrum
G = 1.1
max_A_lim = 10.				# signal-to-noise ratio above which the signal is detected significantly enough
NH = 5000				# number of data points used to construct the Hanning filter
zero_padding_factor = 10		# zero-padding the spectrum to increase the future resolution in the frequency domain: factor corresponding to the number of times we increase
					# the length of the power spectrum by adding zeros
Dnu_max_to_test = 19.			# maximum Dnu value to search for 
period_boundary_distance = 0.1



### Detection of oscillations: estimating the frequency regular spacing, i.e. the large separation Dnu

## Reading the power spectrum: power spectral density versus frequency

path = './spectre_*.txt'
files = sorted(glob.glob(path))
files_number = np.size(files, axis=0)
list_nu_max = np.linspace(10, 270, 18)		# defining the list of nu_max values that will be used to window the spectrum to search for Dnu; the optimal window depends on the physical
						# parameter nu_max, which we do not know a priori
list_Dnu = 0.28 * (list_nu_max)**0.75		# defining the Dnu values to test to window the spectrum, based on a scaling relation with nu_max
size_list_Dnu = np.size(list_Dnu)
for ind_file in range(files_number):		# loop on the number of files to analyze
	start = time.time()			# saving the starting time
	star = files[ind_file]
	data = np.loadtxt(star, skiprows=3)
	freq = data[:,0]				# retrieving the frequencies
	P = data[:,1]					# retrieving the power spectral densities
	list_all_Dnu_final = []
	list_all_EACF_final = []
	list_crit_high_EACF_final = [] 
	list_all_Hanning_centers_final = []
	for i in range(size_list_Dnu):						# loop on the number of Dnu values to test to window the spectrum
		Dnu = list_Dnu[i]
		nu_max = list_nu_max[i]
		dnuH = 1.05 * 2.08 * nu_max**0.15 * Dnu				# defining the width of the Hanning filter
		if nu_max > np.min(freq) + Dnu and nu_max < np.max(freq) - Dnu:	# ensuring that we are not too close to the frequency boundaries of the power spectrum
		
		
			## Windowing the power spectrum with a Hanning filter
		
			freq_corr, spectrum = function_windowing_spectrum_Hanning(NH, nu_max, dnuH, freq, P)


			## Computing the Envelope Autocorrelation Function (EACF) of the power spectrum and taking the maximum value to estimate the frequency regular spacing, i.e. the large separation Dnu
			## and its associated EACF value

			N_spectrum = spectrum.size						# computing the number of data points in the power spectrum
			tau_min = 2. * 10.**6 /(Dnu * G * 3600)					# minimum period boundary to search for Dnu
			tau_max = 2. * 10.**6 /(Dnu / G * 3600)					# maximum period boundary to search for Dnu
			Dnu_found_ini, EACF_Dnu, tau_Dnu, tau_EACF, EACF_star, dnu = function_EACF(tau_min, tau_max, spectrum, zero_padding_factor, freq_corr, N_spectrum, N_oversampling)


			## Appending the Dnu values, associated EACF values, and center frequencies of the Hanning spectrum for each windowed spectrum tested

			if abs(tau_Dnu - np.min(tau_EACF)) > period_boundary_distance and abs(tau_Dnu - np.max(tau_EACF)) > period_boundary_distance:		# ensuring that the Dnu value estimated is far
																			# enough from the period boundaries tested
				list_all_Dnu_final.append(Dnu_found_ini[0])
				list_all_EACF_final.append(EACF_Dnu[0])
				list_all_Hanning_centers_final.append(nu_max) 
	all_Dnu_final = np.array(list_all_Dnu_final)
	all_EACF_final = np.array(list_all_EACF_final)
	all_Hanning_centers_final = np.array(list_all_Hanning_centers_final)


	## Measuring Dnu, its corresponding Envelope Autocorrelation Function values, the center of the Hanning filter used to window the spectrum
			 
	Dnu_final, EACF_final, Hanning_centers_final = function_Dnu_measurement(all_Dnu_final, all_EACF_final, all_Hanning_centers_final)
	
					
	## Computing the uncertainty associated with Dnu

	dnuH_final = 1.05 * 2.08 * Hanning_centers_final**0.15 * Dnu_final				# defining the width of the Hanning filter
	delta_Dnu_over_Dnu = 0.763/(2*np.pi) * Dnu_final/dnuH_final * 100	# precision on Dnu, in percent
	sigma_Dnu = delta_Dnu_over_Dnu * Dnu_final / 100			# uncertainty on Dnu



	#### Plotting the raw and windowed spectrum associated with the final Dnu value measured


	nu_max_plot_ini = Hanning_centers_final[0]				# defining the center of the Hanning filter				
	Dnu_plot = 0.28 * nu_max_plot_ini**0.75	 				# defining the Dnu value used to window the spectrum, based on a scaling relation with nu_max 	
	dnuH_plot = dnuH_final[0]						# defining the width of the Hanning filter
	freq_corr_plot, spectrum_plot = function_windowing_spectrum_Hanning(NH, nu_max_plot_ini, dnuH_plot, freq, P)	# windowing the power spectrum with a Hanning filter
	N_plot_spectrum = spectrum_plot.size				# computing the number of data points in the power spectrum
	tau_min_plot = 2. * 10.**6 /(Dnu_plot * G * 3600)			# minimum period boundary to search for Dnu
	tau_max_plot = 2. * 10.**6 /(Dnu_plot / G * 3600)			# maximum period boundary to search for Dnu
	Dnu_found_plot, EACF_Dnu_plot, tau_Dnu_plot, tau_EACF_plot, EACF_star_plot, dnu_plot = function_EACF(tau_min_plot, tau_max_plot, spectrum_plot, zero_padding_factor, freq_corr_plot, N_plot_spectrum, N_oversampling)									# computing the EACF and estimating Dnu and its associated EACF value
	name_fic = star.replace('./spectre_', '') 
	name_fic = name_fic.replace('.txt','')
	name_fic = str(int(name_fic))
	plt.figure()
	plt.plot(freq, P, label='Raw power spectrum')
	plt.plot(freq_corr_plot, spectrum_plot, c='orange', label='Power spectrum windowed by a Hanning filter')
	plt.yscale('log')
	plt.xlim(np.min(freq), np.max(freq))
	plt.axvline(Hanning_centers_final, color='k', linewidth=1, linestyle ='--' , label= 'Center of the Hanning filter')
	plt.xlabel(r'$\nu$' + ' (' + r'$\mu$' + 'Hz)', fontsize='x-large')
	plt.ylabel(r'$P$' + ' (ppm' + r'$^2/\mu$' 'Hz)', fontsize='x-large')
	plt.legend()
	plt.title('KIC ' + name_fic, fontsize = 'x-large')
	plt.savefig('./Windowed_spectrum_KIC_' + name_fic + '.pdf', format='pdf')
	plt.close()
	
	
	## Plotting the EACF signal versus period

	plt.figure()
	plt.plot(tau_EACF_plot, EACF_star_plot)
	plt.axvline(tau_Dnu_plot, color='k', linewidth=1, linestyle ='--' , label= 'Period corresponding\nto the detected\n' + r'$\Delta \nu$' + ' value')
	plt.xlabel('Period ' + r'$\tau$' + ' (h)', fontsize='x-large')
	plt.ylabel('EACF signal ' + r'$A^\star / \sigma_H$', fontsize='x-large')
	plt.legend()
	plt.title('KIC ' + name_fic, fontsize = 'x-large')
	plt.savefig('./EACF_signal_KIC_' + name_fic + '.pdf', format='pdf')
	plt.close()
	
	
	## Plotting the EACF signals versus the Dnu values found

	plt.figure()
	plt.scatter(all_Dnu_final, all_EACF_final, label=r'$\Delta \nu$' + ' found for each tested part\nof the power spectrum')
	plt.hlines(max_A_lim, 0, 26, linestyle='--', color='grey', linewidth=1, label = 'Detection limit: rejecting\nthe null hypothesis to the 1% level')
	plt.axvline(Dnu_final, linewidth=1, color='k', label='Final ' + r'$\Delta \nu$' + ' found')
	plt.yscale('log')
	plt.title('KIC ' + name_fic, fontsize = 'x-large')
	plt.xlim(0, 20)
	plt.xlabel(r'$\Delta \nu \, (\mu Hz)$', fontsize='x-large')
	plt.ylabel('EACF signals ' + r'$A^\star / \sigma_H$', fontsize='x-large')
	plt.legend(loc=1)
	plt.savefig('./EACF_signals_versus_Dnu_KIC_' + name_fic + '.pdf', format='pdf')
	plt.close()

	
	## Print information
	
	print(str(ind_file + 1) + '. KIC ' + name_fic)
	if EACF_final[0] > max_A_lim:
		print('Oscillations detected: Delta_nu = ' + str('%.3f' % Dnu_final[0]) + ' \u00B1 ' + str('%.3f' % sigma_Dnu[0]) + ' muHz')
	if EACF_final[0] < max_A_lim:
		print('No large frequency separation due to oscillations detected')
	end = time.time()	# saving the ending time
	print('Duration of the run: ' + str('%.3f' % (end - start)) + ' s\n')

