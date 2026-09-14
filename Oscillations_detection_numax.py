### Create the appropriate environment with anaconda: conda env create -f environment.yml
### python=3.9.25, ipython=8.15


### This program aims at detecting oscillations in stars due to internal standing waves, by detecting the corresponding power excess in the frequency power spectrum and measuring the frequency of maximum oscillation power, i.e. nu_max, and its uncertainty, a fundamental parameter needed for stellar physics 

import numpy as np
from scipy.fftpack import fft
import scipy.optimize
import scipy.stats
import pylab as plt
import glob
import time



### Function describing the oscillations: Gaussian

def Gaussian_function(x, amplitude, mean, stddev):
	return amplitude * np.exp(-(x - mean)**2 / stddev**2)



### Routine building an average smoothed power spectrum by performing different smoothings and taking the median value: convolution with Gaussians of different widths

def function_smooth_spectrum(N_smoothing, freq, P):
	array_smooth = np.zeros((N_smoothing,freq.size))		# defining the array that will contain the different smoothed spectra
	list_nu_max_smoothing = np.linspace(30, 270, N_smoothing)	# defining the list of nu_max values that will be used to calibrate the smoothing; the optimal smoothing depends on the physical parameter nu_max, which we want to measure and do not know a priori
	for i in range(list_nu_max_smoothing.size):			# loop on the number of smoothings that are tested
		nu_max_smoothing = list_nu_max_smoothing[i]
		Dnu_smoothing = 0.28 * nu_max_smoothing**0.75		# defining the large separation Dnu, which corresponds to the frequency spacing between consecutive oscillation modes, based on a scaling relation with nu_max
		dnu_env_smoothing = 3*Dnu_smoothing
		sigma_gauss_smoothing = dnu_env_smoothing / (2 * np.sqrt(2. * np.log(2)))
		gaussian_smoothing = Gaussian_function(freq, 1., nu_max_smoothing, sigma_gauss_smoothing)		# defining the Gaussian that will be used to smooth the spectrum
		freq_gaussian_smoothing = freq[np.where(freq >= nu_max_smoothing - dnu_env_smoothing)[0]]		# selecting only the frequencies around the center of the Gaussian  
		gaussian_smoothing = gaussian_smoothing[np.where(freq >= nu_max_smoothing - dnu_env_smoothing)[0]]
		freq_gaussian_smoothing = freq_gaussian_smoothing[np.where(freq_gaussian_smoothing <= nu_max_smoothing + dnu_env_smoothing)[0]]
		gaussian_smoothing = gaussian_smoothing[np.where(freq_gaussian_smoothing <= nu_max_smoothing + dnu_env_smoothing)[0]]
		smooth_i = np.convolve(gaussian_smoothing/gaussian_smoothing.sum(), P, mode='same')						# smoothing the spectrum: convolution with the Gaussian
		array_smooth[i,:] = smooth_i				# saving the smoothed spectrum
	smooth_median = np.median(array_smooth, axis=0)			# building an average smoothed power spectrum: taking the median value of the different smoothed spectra
	return smooth_median



### Routine finding all local maxima in the power spectrum: change of the sign of the difference between consecutive smoothed power spectral densities

def function_find_all_local_maxima(freq, smooth_median):
	diff_sign_diff_smooth_median = np.diff(np.sign(np.diff(smooth_median)))			# computing the difference between consecutive smoothed power spectral densities
	change_sign_crit = np.where(np.sign(diff_sign_diff_smooth_median) != 0)[0]		# finding all the local maxima and minima
												# where the difference between consecutive smoothed power spectral densities changes sign
	freq_change_sign = freq[change_sign_crit]
	smooth_median_change_sign = smooth_median[change_sign_crit]
	crit_maxima = []
	for i in range(freq_change_sign.size):							# loop on all the local maxima and minima
		if smooth_median[change_sign_crit[i]-1] < smooth_median_change_sign[i]:		# finding all local maxima: the sign of the difference between consecutive smoothed power spectral densities
												# has to be positive at frequencies below the local maxima
			crit_maxima.append(change_sign_crit[i])
	freq_change_sign = freq[crit_maxima]
	smooth_median_local_maxima = smooth_median[crit_maxima]
	return freq_change_sign, smooth_median_local_maxima



### Routine eliminating borders effects: if the minimum frequency at which there is a local minimum is zero, cut all frequencies in the vicinity as well as all frequencies in a range of same width
### which upper limit corresponds to the highest frequency	

def function_eliminating_border_effects(freq, smooth_median, freq_change_sign, freq_cut_factor):
	crit_range_min = np.where(freq <= freq_change_sign[0])[0]
	diff_range_min = np.diff(smooth_median[crit_range_min])
	local_minimum_crit = np.where(diff_range_min > 0)[0][0]		# finding the minimum frequency at which there is a local minimum
	list_border_effect = []
	if local_minimum_crit == 0: 					# case where the minimum frequency at which there is a local minimum is zero
		list_border_effect.append(freq_change_sign[0])
		crit_lower_cut = np.where(freq >= list_border_effect[0] + freq_cut_factor * list_border_effect[0])[0]
		crit_higher_cut = np.where(freq <= np.max(freq) - (list_border_effect[0] + freq_cut_factor * list_border_effect[0]))[0]
		smooth_median = smooth_median[crit_lower_cut]		# cutting the lowest frequencies
		freq = freq[crit_lower_cut]
		smooth_median = smooth_median[crit_higher_cut]		# cutting the highest frequencies
		freq = freq[crit_higher_cut]
		return freq, smooth_median



### Routine finding the most significant local maxima: in a frequency range around the position of the maximum, the median value of the difference between consecutive smoothed power spectral densities
### has to be positive at frequencies lower than the position of the maximum and negative above

def function_finding_main_local_maxima(freq_change_sign, freq, smooth_median):
	nu_max_test = freq_change_sign				# considering each local maximum as a nu_max test value
	Dnu_test =  0.28 * nu_max_test**0.75			# defining a test Dnu value based on a scaling relation with each nu_max test value
	dnu_env_test = 0.59 * nu_max_test**0.90			# defining a width around each nu_max test value to look for the most significant local maxima
	main_max_freq = []
	main_max_smooth_median = []
	for i in range(nu_max_test.size):			# loop on all the local maxima
		crit_range_left = np.where(freq >= nu_max_test[i] - dnu_env_test[i]/4)[0]		# selecting the lower frequency range around each local maximum
		freq_range_left = freq[crit_range_left]
		P_range_left = smooth_median[crit_range_left]
		crit_range_left = np.where(freq_range_left <= nu_max_test[i])[0]
		freq_range_left = freq_range_left[crit_range_left]
		P_range_left = P_range_left[crit_range_left]
		crit_range_right = np.where(freq > nu_max_test[i])[0]					# selecting the upper frequency range around each local maximum
		freq_range_right = freq[crit_range_right]
		P_range_right = smooth_median[crit_range_right]
		crit_range_right = np.where(freq_range_right <= nu_max_test[i] + dnu_env_test[i]/2)[0]
		freq_range_right = freq_range_right[crit_range_right]
		P_range_right = P_range_right[crit_range_right]
		diff_range_left = np.diff(P_range_left)		# computing the difference between consecutive smoothed power spectral densities in the lower frequency range around each local maximum
		diff_range_right = np.diff(P_range_right)	# computing the difference between consecutive smoothed power spectral densities in the upper frequency range around each local maximum

		if np.median(diff_range_left) > 0 and np.median(diff_range_right) < 0:		# case where the median value of the difference between consecutive smoothed power spectral densities is
												# positive in the lower frequency range and negative in the upper frequency range around each local maximum
			main_max_freq.append(nu_max_test[i])
			main_max_smooth_median.append(smooth_median_local_maxima[i])
	main_max_freq = np.array(main_max_freq)							# saving the most significant local maxima
	main_max_smooth_median = np.array(main_max_smooth_median)				# saving the smoothed power spectral densities associated with the most significant local maxima
	return main_max_freq, main_max_smooth_median



### Routine gathering the main maxima into clusters: minimizing the number of nu_max test values to try when fitting the oscillations

def function_cluster_gathering(main_max_freq, freq_dist, main_max_smooth_median):
	crit_clusters = np.where(np.diff(main_max_freq) > freq_dist)[0]
	if list(crit_clusters):		# case where the data points that are spaced enough one from another
		clusters_nu = np.zeros((crit_clusters.size+1))
		clusters_nu[:crit_clusters.size] = main_max_freq[crit_clusters]
		clusters_nu[crit_clusters.size:] = main_max_freq[crit_clusters+1][crit_clusters.size-1]
	else:				# case where the data points that are too close one from another: grouping them by taking the median of their frequencies
		clusters_nu = np.array([np.median(main_max_freq)])
	list_cluster_nu_max = [] 
	list_cluster_PSD = []
	crit_cluster_i = np.where(main_max_freq <= clusters_nu[0])[0]		# selecting the frequencies and smoothed power spectral densities of the main maxima that fall below the cluster
										# with lowest frequency
	nu_max_cluster_i = main_max_freq[crit_cluster_i]
	PSD_cluster_i = main_max_smooth_median[crit_cluster_i]
	list_cluster_nu_max.append(np.median(nu_max_cluster_i))			# grouping together the frequencies and smoothed power spectral densities of the main maxima that fall below the cluster
										# with lowest frequency by taking their median value 
	list_cluster_PSD.append(np.median(PSD_cluster_i))
	for i in range(1,clusters_nu.size):					# loop on the number of clusters
		crit_cluster_i = np.where(main_max_freq <= clusters_nu[i])[0]	# selecting the frequencies and smoothed power spectral densities of the main maxima that fall between two clusters
		nu_max_cluster_i = main_max_freq[crit_cluster_i]
		PSD_cluster_i = main_max_smooth_median[crit_cluster_i]
		crit_cluster_i = np.where(nu_max_cluster_i > clusters_nu[i-1])[0]
		nu_max_cluster_i = nu_max_cluster_i[crit_cluster_i]
		PSD_cluster_i = PSD_cluster_i[crit_cluster_i]
		list_cluster_nu_max.append(np.median(nu_max_cluster_i))		# grouping together the frequencies and smoothed power spectral densities of the main maxima that fall between two clusters
										# by taking their median value 
		list_cluster_PSD.append(np.median(PSD_cluster_i))
	return list_cluster_nu_max, list_cluster_PSD



### Function describing the background signal locally around the oscillations: power law

def local_background_function(x, mean, a, b):
	return a * (x/mean)**b


	
### Function describing locally the oscillations to be fitted: a Gaussian envelope plus a power law for the background signal

def oscillations_function(x, amplitude, mean, stddev, a, b):
	return Gaussian_function(x, amplitude, mean, stddev) + local_background_function(x, mean, a, b)	
	


## Routine fitting the oscillations
		
def function_fitting_oscillations(freq, nu_max_test, dnu_env_test, variation_factor, freq_local, smooth_local):
	crit_nu_max_to_test = np.where(freq >= nu_max_test - 1. * dnu_env_test)[0]		# selecting the frequency range around each nu_max test value to fit the oscillations
	nu_max_to_test = freq[crit_nu_max_to_test]
	crit_nu_max_to_test = np.where(nu_max_to_test <= nu_max_test + 1. * dnu_env_test)[0]
	nu_max_to_test = nu_max_to_test[crit_nu_max_to_test]
	min_nu_max_to_test = nu_max_to_test[0]					# defining the minimum frequency boundary to test
	max_nu_max_to_test = nu_max_to_test[nu_max_to_test.size-1]		# defining the maximum frequency boundary to test
	sigma_gauss_test_min = (dnu_env_test - variation_factor * dnu_env_test) / (2 * np.sqrt(2. * np.log(2)))		# defining the minimum width of the Gaussian envelope to test
	sigma_gauss_test_max = (dnu_env_test + variation_factor*dnu_env_test) / (2 * np.sqrt(2. * np.log(2)))		# defining the maximum width of the Gaussian envelope to test
	params_local, cov_local = scipy.optimize.curve_fit(oscillations_function, freq_local, smooth_local, bounds=([PSD_test/15, min_nu_max_to_test + dnu_env_test/3, sigma_gauss_test_min, 0, -5.],[15.*PSD_test, max_nu_max_to_test - dnu_env_test/3, sigma_gauss_test_max, np.max(smooth_local), 0.]))							# performing the fit: obtaining the oscillations parameters
	oscillations_test = oscillations_function(freq_local, params_local[0], params_local[1], params_local[2], params_local[3], params_local[4])	# computing the oscillations
	return params_local, cov_local, oscillations_test



### Routine computing the ratio between the smoothed power spectral density at nu_max and the background

def function_computing_oscillations_amplitude(freq, params_local, width_factor, smooth_median, freq_local, smooth_local, local_B, list_amplitude_possible_Gaussian, list_amplitude_oscillations_photon_noise, list_nu_max_possible):

	## Estimating the white noise level: at high frequency, significantly above the frequency range of oscillations
		
	dnu_env_local = 0.59 * params_local[1]**0.90
	photon_noise_crit = np.where(freq > params_local[1] + width_factor * dnu_env_local)[0]
	if list(photon_noise_crit):
		mean_photon_noise = np.median(smooth_median[photon_noise_crit])
	else:
		mean_photon_noise = np.median(smooth_median[np.where(freq >= freq_lower_limit_white_noise)[0]])


	## Computing the Gaussian amplitude relative to the local background at nu_max and relative to the white noise level
	
	crit_nu_max_test = np.where(abs(params_local[1] - freq_local) == np.min(abs(params_local[1] - freq_local)))[0]
	if crit_nu_max_test.size > 1:
		crit_nu_max_test = crit_nu_max_test[0]
	amplitude_gaussian_median_smoothed = (smooth_local[crit_nu_max_test] - local_B[crit_nu_max_test]) / local_B[crit_nu_max_test] * 100		# computing the Gaussian amplitude relative to the local background at nu_max, in percent
	amplitude_gaussian_photon_noise = (smooth_local[crit_nu_max_test] - mean_photon_noise) / smooth_local[crit_nu_max_test] * 100			# computing the Gaussian amplitude relative to the white noise level, in percent
	list_amplitude_possible_Gaussian.append(amplitude_gaussian_median_smoothed)
	list_amplitude_oscillations_photon_noise.append(amplitude_gaussian_photon_noise)
	list_nu_max_possible.append(params_local[1])
	return

	
	
### Routine selecting the optimal nu_max to fit of oscillations: highest ratio between the smoothed power spectral density at nu_max and the background
	
def function_selecting_optimal_fit_oscillations(nu_max_estimate, amplitude_oscillations_photon_noise, ratio_oscillations_white_noise, amplitude_Gaussian_estimate):
	crit_nu_max_estimate = np.where(amplitude_oscillations_photon_noise >= ratio_oscillations_white_noise)[0]		# ensuring that the Gaussian is significantly enough above the white noise level
	if list(crit_nu_max_estimate):	
		nu_max_estimate = nu_max_estimate[crit_nu_max_estimate]	
		amplitude_Gaussian_estimate = amplitude_Gaussian_estimate[crit_nu_max_estimate]	
	if list(amplitude_Gaussian_estimate):	
		crit_nu_max_estimate = np.where(amplitude_Gaussian_estimate == np.max(amplitude_Gaussian_estimate))[0]		# selecting the optimal configuration
		nu_max_estimate = nu_max_estimate[crit_nu_max_estimate]
	else:									# case where configuration matches the detection criteria
		nu_max_estimate = np.array([np.median(freq)])			# defining an estimate of nu_max as the median of the entire frequency range of the power spectrum
	return nu_max_estimate



### Routine refining the measurement of nu_max: taking the frequency in the power spectrum that is close to nu_max and with highest smoothed power spectral density

def function_refining_nu_max_measurement(freq, params_local, dnu_env_final, smooth_spectrum):
	crit_close_nu_max = np.where(freq >= params_local[1] - dnu_env_final/4)[0]
	freq_close_nu_max = freq[crit_close_nu_max]
	PSD_close_nu_max = smooth_spectrum[crit_close_nu_max]
	crit_close_nu_max = np.where(freq_close_nu_max <= params_local[1] + dnu_env_final/4)[0]
	freq_close_nu_max = freq_close_nu_max[crit_close_nu_max]
	PSD_close_nu_max = PSD_close_nu_max[crit_close_nu_max]
	nu_max_final = freq_close_nu_max[np.where(PSD_close_nu_max == np.max(PSD_close_nu_max))[0]][0]		# refining the measurement of nu_max
	return nu_max_final



### Routine assessing if oscillations are detected or not: comparing the fit with oscillations with respect to a fit without oscillations through the likelihood ratio test

def function_oscillations_detection(local_background_function, freq_local, smooth_local, freq_min_gauss, dnu_env_estimate, freq_max_gauss, smooth_spectrum, oscillations_final, width_factor, freq_lower_limit_white_noise, ratio_oscillations_white_noise):

	## Computing the likelihood ratio
			
	no_params_local, no_cov_local = scipy.optimize.curve_fit(local_background_function, freq_local, smooth_local, bounds=([freq_min_gauss[0] + dnu_env_estimate[0] / 3, 0, -5], [freq_max_gauss[0] - dnu_env_estimate[0] / 3, np.max(smooth_spectrum), 0.]))	# fitting
																																	# only the
																																	# local
																																	# background
																																	# without
																																	# oscillations
	no_oscillations_final = local_background_function(freq_local, no_params_local[0], no_params_local[1], no_params_local[2])	# computing the local background without oscillations
	LL_oscillations = np.sum(scipy.stats.norm.pdf(smooth_local, loc=oscillations_final))						# computing the likelihood of the model with oscillations
	LL_no_oscillations = np.sum(scipy.stats.norm.pdf(smooth_local, loc=no_oscillations_final))					# computing the likelihood of the model without oscillations
	LL_ratio = -2. * np.log(LL_no_oscillations / LL_oscillations)									# computing the likelihood ratio between the models without and with oscillations
	p_value = 1. - scipy.stats.norm.cdf(LL_ratio)											# computing the p-value: probability of obtaining the observed results by assuming that the null hypothesis (no oscillations) is true


	## Estimating the white noise level: at high frequency, significantly above the frequency range of oscillations

	dnu_env_final = 0.59 * params_local[1]**0.90					
	if params_local[1] + width_factor * dnu_env_final < freq_lower_limit_white_noise :	
		mean_photon_noise = np.median(smooth_spectrum[np.where(freq >= freq_lower_limit_white_noise)[0]])
	else:									
		mean_photon_noise = np.median(local_B[np.where(freq_local >= freq_lower_limit_white_noise)[0]])					

				
	## Checking if the Gaussian amplitude is significantly enough above the white noise level
		
	PSD_nu_max = oscillations_final[np.where(abs(freq_local - params_local[1]) == np.min(abs(freq_local - params_local[1])))[0]]		
	PSD_nu_max_back = local_B[np.where(abs(freq_local - params_local[1]) == np.min(abs(freq_local - params_local[1])))[0]]	
	crit_photon_noise = np.where((PSD_nu_max - mean_photon_noise)/ PSD_nu_max * 100 >= ratio_oscillations_white_noise)[0]
	return p_value, crit_photon_noise



### Routine computing the uncertainty associated with nu_max

def function_uncertainty_nu_max(dnu_env_final):
		sigma_env_final = dnu_env_final / (2. * np.sqrt(2. * np.log(2)))
		nu_res = 1./(4. * 365. * 24. * 3600) * 10.**6	# frequency resolution in the power spectrum, in muHz
		sigma_nu_max_final = 9. * nu_res * (1. + 4. / (3.84 / sigma_env_final)**(2./3))
		return sigma_nu_max_final



### Definition of needed parameters

N_smoothing = 18
freq_cut_factor = 0.03
freq_dist = 1.
variation_factor = 0.1
width_factor = 1.2
ratio_oscillations_white_noise = 10.
freq_lower_limit_white_noise = 200
p_value_upper_limit = 1.
alpha = 1.



### Detection of oscillations 

## Reading the power spectrum: power spectral density versus frequency

path = './spectre_*.txt'
files = sorted(glob.glob(path))
files_number = np.size(files, axis=0)
for ind_file in range(files_number):		# loop on the number of files to analyze
	start = time.time()			# saving the starting time
	star = files[ind_file]
	data = np.loadtxt(star, skiprows=3)
	freq = data[:,0]				# retrieving the frequencies
	P = data[:,1]					# retrieving the power spectral densities


	### Building an average smoothed power spectrum by performing 18 different smoothings and taking the median value

	smooth_median = function_smooth_spectrum(N_smoothing, freq, P)


	## Finding all local maxima in the smoothed power spectrum

	freq_change_sign, smooth_median_local_maxima = function_find_all_local_maxima(freq, smooth_median)

	
	## Eliminating borders effects	

	freq, smooth_median = function_eliminating_border_effects(freq, smooth_median, freq_change_sign, freq_cut_factor)
								

	## Finding the most significant local maxima

	main_max_freq, main_max_smooth_median = function_finding_main_local_maxima(freq_change_sign, freq, smooth_median)


	## Gathering the main maxima into clusters

	list_cluster_nu_max, list_cluster_PSD = function_cluster_gathering(main_max_freq, freq_dist, main_max_smooth_median)


	## Defining parameters to fit the oscillations 
	
	list_nu_max_possible = []
	list_amplitude_possible_Gaussian = []
	list_amplitude_oscillations_photon_noise = []
	for i in range(np.size(list_cluster_PSD)):		# loop on the number of clusters: number of nu_max test values to try when fitting the oscillations
		nu_max_test = list_cluster_nu_max[i]
		PSD_test = list_cluster_PSD[i]
		Dnu_test =  0.28 * nu_max_test**0.75		# defining a test Dnu value based on a scaling relation with each nu_max test value	
		dnu_env_test = 0.59 * nu_max_test**0.90		# defining a width around each nu_max test value to fit the oscillations
		
		
		## Estimating the local background parameters
		
		freq_min_gauss = nu_max_test - dnu_env_test		# defining the minimum frequency used to obtain a first estimate of the local background parameters
		freq_max_gauss = nu_max_test + dnu_env_test		# defining the maximum frequency used to obtain a first estimate of the local background parameters
		crit_min_gauss = np.where(abs(freq_min_gauss - freq) == np.min(abs(freq_min_gauss - freq)))[0]
		crit_max_gauss = np.where(abs(freq_max_gauss - freq) == np.min(abs(freq_max_gauss - freq)))[0]
		smooth_min_gauss = smooth_median[crit_min_gauss]	# selecting the corresponding smoothed power spectral densities
		smooth_max_gauss = smooth_median[crit_max_gauss]
		b = (np.log10(smooth_min_gauss) - np.log10(smooth_max_gauss)) / (np.log10(freq_min_gauss) - np.log10(freq_max_gauss))	# estimating the local background parameters
		a = (smooth_min_gauss + smooth_max_gauss) / ((freq_min_gauss/nu_max_test)**b + (freq_max_gauss/nu_max_test)**b)
		
		
		## Computing the local background estimate
		
		freq_local = freq[np.where(freq >= nu_max_test - dnu_env_test)[0]]
		smooth_local = smooth_median[np.where(freq >= nu_max_test - dnu_env_test)[0]]
		freq_local = freq_local[np.where(freq_local <= nu_max_test + dnu_env_test)[0]]
		smooth_local = smooth_local[np.where(freq_local <= nu_max_test + dnu_env_test)[0]]
		local_B = a * (freq_local / nu_max_test)**b
		
		
		## Fitting the oscillations and selecting the optimal configuration with highest amplitude of the oscillations

		params_local, cov_local, oscillations_test = function_fitting_oscillations(freq, nu_max_test, dnu_env_test, variation_factor, freq_local, smooth_local)		# fitting the oscillations
		function_computing_oscillations_amplitude(freq, params_local, width_factor, smooth_median, freq_local, smooth_local, local_B, list_amplitude_possible_Gaussian, list_amplitude_oscillations_photon_noise, list_nu_max_possible)		# computing the ratio
																															# between the smoothed power 
																															# spectral density at nu_max
																															# and the background	
	nu_max_estimate = np.array(list_nu_max_possible)
	amplitude_Gaussian_estimate = np.array(list_amplitude_possible_Gaussian)
	amplitude_oscillations_photon_noise = np.array(list_amplitude_oscillations_photon_noise)
	nu_max_estimate = function_selecting_optimal_fit_oscillations(nu_max_estimate, amplitude_oscillations_photon_noise, ratio_oscillations_white_noise, amplitude_Gaussian_estimate)	# selecting the optimal nu_max to fit of oscillations: highest ratio between
																								# the smoothed power spectral density at nu_max and the background
																								
																								
	## Computing the optimally smoothed spectrum

	freq = data[:,0]													# working again with the entire frequency range of the power spectrum
	Dnu_estimate = 0.28 * nu_max_estimate**0.75										# estimating Dnu based on a scaling relation with the estimated optimal nu_max to fit the oscillations
	dnu_env_convolve_estimate = 3*Dnu_estimate										# defining a width around the estimated optimal nu_max to fit the oscillations
	sigma_gauss_estimate = dnu_env_convolve_estimate / (2 * np.sqrt(2. * np.log(2)))	
	gaussian = Gaussian_function(freq, 1., nu_max_estimate, sigma_gauss_estimate)						# defining the Gaussian that will be used to smooth the spectrum
	freq_gaussian = freq[np.where(freq >= nu_max_estimate - dnu_env_convolve_estimate)[0]]					# selecting only the frequencies around the center of the Gaussian
	gaussian = gaussian[np.where(freq >= nu_max_estimate - dnu_env_convolve_estimate)[0]]
	freq_gaussian = freq_gaussian[np.where(freq_gaussian <= nu_max_estimate + dnu_env_convolve_estimate)[0]]
	gaussian = gaussian[np.where(freq_gaussian <= nu_max_estimate + dnu_env_convolve_estimate)[0]]
	smooth_spectrum = np.convolve(gaussian/gaussian.sum(), P, mode='same')							# smoothing the spectrum: convolution with the Gaussian
	PSD_estimate = smooth_spectrum[np.where(abs(freq - nu_max_estimate) == np.min(abs(freq - nu_max_estimate)))[0]]		# estimating the amplitude of the Gaussian: smoothed power spectral density at the estimated optimal nu_max


	## Finding all local maxima in the smoothed power spectrum
			
	freq_change_sign, smooth_change_sign = function_find_all_local_maxima(freq, smooth_spectrum)


	## Eliminating borders effects		

	freq, smooth_spectrum = function_eliminating_border_effects(freq, smooth_spectrum, freq_change_sign, freq_cut_factor)


	## Defining parameters to fit the oscillations 

	dnu_env_estimate = 0.59 * nu_max_estimate**0.90		# defining a width around the optimal nu_max to fit of oscillations


	## Estimating the local background parameters

	freq_min_gauss = nu_max_estimate - width_factor * dnu_env_estimate		# defining the minimum frequency used to obtain a first estimate of the local background parameters
	freq_max_gauss = nu_max_estimate + width_factor * dnu_env_estimate		# defining the maximum frequency used to obtain a first estimate of the local background parameters
	crit_min_gauss = np.where(abs(freq_min_gauss - freq) == np.min(abs(freq_min_gauss - freq)))[0]
	crit_max_gauss = np.where(abs(freq_max_gauss - freq) == np.min(abs(freq_max_gauss - freq)))[0]
	smooth_min_gauss = smooth_spectrum[crit_min_gauss]				# selecting the corresponding smoothed power spectral densities
	smooth_max_gauss = smooth_spectrum[crit_max_gauss]
	b = (np.log10(smooth_min_gauss) - np.log10(smooth_max_gauss)) / (np.log10(freq_min_gauss) - np.log10(freq_max_gauss))		# estimating the local background parameters
	a = (smooth_min_gauss + smooth_max_gauss) / ((freq_min_gauss/nu_max_estimate)**b + (freq_max_gauss/nu_max_estimate)**b)
	
	
	## Computing the local background estimate
	
	freq_local = freq[np.where(freq >= nu_max_estimate - width_factor * dnu_env_estimate)[0]]
	smooth_local = smooth_spectrum[np.where(freq >= nu_max_estimate - width_factor * dnu_env_estimate)[0]]
	freq_local = freq_local[np.where(freq_local <= nu_max_estimate + width_factor * dnu_env_estimate)[0]]
	smooth_local = smooth_local[np.where(freq_local <= nu_max_estimate + width_factor * dnu_env_estimate)[0]]
	local_B_ini = a * (freq_local / nu_max_estimate)**b		


	## Fitting the oscillations

	sigma_gauss_estimate_min = (dnu_env_estimate - variation_factor * dnu_env_estimate)  / (2 * np.sqrt(2. * np.log(2)))				# defining the minimum width of the Gaussian envelope to fit
	sigma_gauss_estimate_max = (dnu_env_estimate + variation_factor * dnu_env_estimate) / (2 * np.sqrt(2. * np.log(2)))				# defining the maximum width of the Gaussian envelope to fit
	params_local, cov_local = scipy.optimize.curve_fit(oscillations_function, freq_local, smooth_local, bounds=([PSD_estimate[0]/15, freq_min_gauss[0] + dnu_env_estimate[0]/3, sigma_gauss_estimate_min[0], 0, -5],[15.*PSD_estimate[0], freq_max_gauss[0] - dnu_env_estimate[0]/3, sigma_gauss_estimate_max[0], np.max(smooth_spectrum), 0.]))												# performing the fit: obtaining the oscillations parameters
	oscillations_final = oscillations_function(freq_local, params_local[0], params_local[1], params_local[2], params_local[3], params_local[4])	# computing the oscillations
	local_B = params_local[3] * (freq_local/params_local[1])**params_local[4]									# computing the final local background


	## Plotting the smoothed power spectrum with the optimal local fit of oscillations

	name_fic = star.replace('./spectre_', '') 
	name_fic = name_fic.replace('.txt','')
	name_fic = str(int(name_fic))
	plt.figure()
	plt.plot(freq, smooth_spectrum)
	plt.plot(freq_local, oscillations_final, color='orange', label='Local fit of oscillations')
	plt.xlabel(r'$\nu$' + ' (' + r'$\mu$' + 'Hz)', fontsize='x-large')
	plt.ylabel(r'$P$' + ' (ppm' + r'$^2/\mu$' 'Hz)', fontsize='x-large')
	plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
	plt.title('KIC ' + name_fic, fontsize='x-large')
	plt.xlim(np.min(freq), np.max(freq))
	plt.legend()
	plt.savefig('./Smoothed_spectrum_fit_KIC_' + name_fic + '.pdf', format='pdf')
	plt.close()


	## Assessing if oscillations are detected or not: comparing the fit with oscillations with respect to a fit without oscillations through the likelihood ratio test
	
	p_value, crit_photon_noise = function_oscillations_detection(local_background_function, freq_local, smooth_local, freq_min_gauss, dnu_env_estimate, freq_max_gauss, smooth_spectrum, oscillations_final, width_factor, freq_lower_limit_white_noise, ratio_oscillations_white_noise)								# computing the likelihood ratio and its p-value
	print(str(ind_file + 1) + '. KIC ' + name_fic)
	if p_value * 100 < alpha and list(crit_photon_noise):				# likelihood ratio test: case where oscillations are detected
		dnu_env_final = 0.59 * params_local[1]**0.90
		nu_max_final = function_refining_nu_max_measurement(freq, params_local, dnu_env_final, smooth_spectrum)		# refining the measurement of nu_max
		sigma_nu_max_final = function_uncertainty_nu_max(dnu_env_final)		# computing the uncertainty associated with nu_max
		

		## Potting the raw power spectrum with oscillations identified by the position of the nu_max frequency
		
		freq_raw = data[:,0]
		P_raw = data[:,1]
		plt.figure()
		plt.plot(freq_raw, P_raw)
		plt.axvline(nu_max_final, color='k', linewidth=2, label= r'$\nu_{max}$' + ' measured')
		plt.yscale('log')
		plt.xlabel(r'$\nu$' + ' (' + r'$\mu$' + 'Hz)', fontsize='x-large')
		plt.ylabel(r'$P$' + ' (ppm' + r'$^2/\mu$' 'Hz)', fontsize='x-large')
		plt.title('KIC ' + name_fic, fontsize='x-large')
		plt.xlim(np.min(freq_raw), np.max(freq_raw))
		plt.ylim(np.min(P_raw)/2, 20*np.max(P_raw))
		plt.legend()
		plt.savefig('Spectrum_with_nu_max_KIC_' + name_fic + '.pdf', format='pdf')
		plt.close()
		print('Oscillations detected: nu_max = ' + str('%.3f' % nu_max_final) + ' \u00B1 ' + str('%.3f' % sigma_nu_max_final) + ' muHz')	
	else:								# likelihood ratio test: case where no oscillations are detected
		print('No power escess due to oscillations detected')
	end = time.time()	# saving the ending time
	print('Duration of the run: ' + str('%.3f' % (end - start)) + ' s\n')

