### Project: Oscillations_detection_numax.py

I completed this project during my postdoc in Porto, Portugal (2019 - 2021).


### Project overview

This project aims at detecting oscillations in stars due to the propagation of internal waves inside stars, by detecting the corresponding power excess in the frequency power spectrum and measuring the frequency of maximum oscillation power, i.e. nu_max, and its uncertainty. The detection of oscillations allows us to probe the interior of stars, a field called asteroseismology, i.e. stellar seismology that works on a principle similar as for Earth's seismology. Measuring nu_max enables to estimate the mass and radius of stars through physical scaling relations, which are fundamental parameters to constrain stellar physics.


### Dataset

The dataset is composed of power spectra obtained beforehand by taking the Fourier transform of the light curves of stars, i.e. flux time-series; indeed, the oscillation modes generate extremely faint and periodic variations of the flux of stars, making possible to identify the oscillation modes by their frequency in the power spectrum. The dataset is composed of power spectra for about 3900 red giant stars observed by the space missions Kepler and TESS, i.e. one power spectrum per star. Here, only one power spectrum obtained by the Kepler spacce mission is provided as an example: spectre_*.txt.


### Methodology

## Data preprocessing

Data preprocessing steps involve smoothing the power spectrum to ease the search for the Gaussian envelope of the stellar oscillations. The optimal degree of smoothing is not known a priori, hence 20 different smoothings are applied by convolving the power spectrum with a Gaussian. A composite smoothed spectrum is then built by taking the median of the power spectral density among all the 20 smoothed spectra. All the significant local maxima are then found.


## Model

Two models are used. The first model is composed of a Gaussian envelope for the stellar oscillations and of a local contribution for the background signal under the form of a power law with frequency. The second a model includes only the local contribution of the background and no stellar oscillations.


## Fitting the model to the data

The frequency of each significant local maximum is iteratively taken as a first estimate for the center of the Gaussian. The first model including stellar oscillations is then fitted to the data in a limited frequency range around the frequency of each local maximum, by exploring a range of values for the 5 following parameters: the center frequency, the amplitude and the width of the Gaussian; the amplitude and the exponent of the power law describing the background contribution. The best fit is then selected as the configuration maximizing the relative variation between the smoothed power spectral density and the local background at the center frequency of the Gaussian, ensuring maximal height-to-background ratio for the stellar oscillations. The second model including only the local contribution of the background and no stellar oscillations is then fitted to the data.


### Measurement

The significance of the two models (with and without stellar oscillations) with respect to the data is finally compared through the likelihood ratio test. This hypothesis test compares the goodness-of-fit of two models to determine which offers a better fit to the data; the null hypothesis is that the data is better fitted by the first model with fewer parameters, i.e. without stellar oscillations. The likelihood ratio approximately follows a chi-square distribution under the null hypothesis; a chi-square statistics with 2 degrees of freedom is used because the
model with oscillations includes 2 additional parameters compared to the model without oscillations. Oscillations are considered signigcantly detected if the p-value, i.e. the probability of obtaining the observed results by assuming that the null hypothesis is true, is below 1%. The frequency of maximum oscillation power, i.e. nu_max, then corresponds to the central frequency of the Gaussian used to fit the oscillations. The 1-sigma uncertainty on nu_max is determined through a physical scaling relation involving the duration over which the star was observed to obtain the power spectrum, as well as the ratio between the height of the Gaussian envelope for stellar oscillations and the local background.


### Results

The results are printed in the terminal: whether stellar oscillations are detected or not, and the values of nu_max and its 1-sigma uncertainty in the case of a detection. The raw power spectrum, i.e. power spectral density as a function frequency, is saved as Spectrum_with_nu_max_*.pdf; the frequency corresponding to the measured nu_max is represented by the vertical black line. The optimally smoothed power spectrum is saved as Smoothed_spectrum_fit_*.pdf; the best fit for stellar oscillations including the contribution of the local background is represented in orange.


### Key findings

This project led to the publication of a scientific article in the international review Astronomical Notes in 2021: Gehan et al. 2021, A&A, vol. 344, 5. The two plots presented here were selected to make the cover of the issue number of the journal.


### Installation: with anaconda

git clone https://github.com/cgehan-astro/Postdoc_Portugal_oscillations_detection.git
cd Postdoc_Portugal_oscillations_detection
conda env create -f environment.yml
