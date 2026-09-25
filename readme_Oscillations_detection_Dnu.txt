### Project: Oscillations_detection_Dnu.py

I completed this project during my postdoc in Porto, Portugal (2019 - 2021).


### Project overview

This project aims at detecting oscillations in stars due to the propagation of internal waves inside stars, by detecting and measuring the frequency spacing between consecutive oscillation modes in the frequency power spectrum, i.e the large separation Dnu, and its uncertainty. The detection of oscillations allows us to probe the interior of stars, a field called asteroseismology, i.e. stellar seismology that works on a principle similar as for Earth's seismology. Measuring Dnu enables to estimate the mass and radius of stars through physical scaling relations, which are fundamental parameters to constrain stellar physics.


### Dataset

The dataset is composed of power spectra obtained beforehand by taking the Fourier transform of the light curves of stars, i.e. flux time-series; indeed, the oscillation modes generate extremely faint and periodic variations of the flux of stars, making possible to identify the oscillation modes by their frequency in the power spectrum. The dataset is composed of power spectra for about 3900 red giant stars observed by the space missions Kepler and TESS, i.e. one power spectrum per star. Here, only one power spectrum obtained by the Kepler spacce mission is provided as an example: spectre_*.txt.


### Methodology


## Model

The model is composed of a Hanning filters by which the power spetral density is mutliplied in order to select only a portion the power spectrum toperform a blind search for the large separation Dnu.


### Measurement

The power spectrum is multiplied by a Hanning filter, for which 18 different center frequencies are tested, providing us with 18 windowed power spectra. Each of the windowed power spectrum is zero-padded to ensure a high-enough resolution in the period space. The Fourier transform of each of the windowed power spectrum is then computed, which is similar to computing the autocorrelation of the light curve, i.e. flux time-serie from which the power spectrum was obtained beforehand. The Fourier transform is then normalized and divided by the mean noise level in the autocorrelation in order to accurately compare the strength of the signal for each windowed spectrum; the mean noise level in the autocorrelation is estimated through a scaling relation with the number of data points in the zero-padded windowed power spectrum and with the number of data points in the raw power spectrum. For each of the windowed power spectrum, the signature of the large separation corresponds to the first peak in the autocorrelation signal. The final Dnu value is then selected as the one with highest autocorrelation signal. Oscillations are considered significantly detected if the autocorrelation signal associated to Dnu is at least equal to a threshold value corresponding to rejecting the null hypothesis to the 0.4% level; the null hypothesis is that the detected signal can be explained by noise only. The 1-sigma uncertainty on Dnu is determined through a physical scaling relation involving the width of the optimal Hanning filter used.


### Results

The results are printed in the terminal: whether stellar oscillations are detected or not, and the values of Dnu and its 1-sigma uncertainty in the case of a detection. The raw power spectrum, i.e. power spectral density as a function frequency, is saved as Windowed_spectrum_KIC_*.pdf; the optimal frequency of the Hanning filter is represented by the vertical black dashed lines, and the power spectrum windowed by the Hanning filter and used to look for Dnu is represented in orange. The optimal autocorrelation signal is saved as EACF_signal_KIC_*.pdf; the period corresponding to the detected Dnu value is represented by the vertical black dashed lines. The maximum values of the autocorrelation signal computed for different intervals of frequency are saved in EACF_signals_versus_Dnu_KIC_*.pdf as a function of the Dnu values found; the final DNu value detected is represented by a vertical black line, while the detection limit corresponding to the rejection of the null hypothesis to the 1% level is represented by the horizontal grey dashed lines.


### Key findings

This project led to the publication of a scientific article in the international review Astronomical Notes in 2021: Gehan et al. 2021, A&A, vol. 344, 5.


### Installation: with anaconda

git clone https://github.com/cgehan-astro/Postdoc_Portugal_oscillations_detection.git
cd Postdoc_Portugal_oscillations_detection
conda env create -f environment.yml
