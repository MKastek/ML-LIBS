# ML-LIBS

My approach to solve [LIBS 2022 quantification contest](https://libs2022.com/). Task was to predict the Cr, Mn, Mo, and Ni content of 15 metal alloys. 
Every sample in dataset contains LIBS spectra:

![](https://github.com/MKastek/ML-LIBS/blob/f6959529fe8640bdc8797b75db22888687925a59/images/libs_sample_spectrum.PNG?raw=true)  

Two approches of dimension reduction were tested:
- selecting lines with best correlation with composition of given element,
- PCA analysis.

Final model was built with XGBoost Regressor.

## Final result of the contest
![alt text](https://github.com/MKastek/ML-LIBS/blob/f33801ec87d9302a152d33cdfb89cafd17d85a28/images/final_result_rmse.jpeg?raw_data=True)  
[Certification](https://github.com/MKastek/ML-LIBS/blob/master/download/certificationLetter_MarcinKastek_signed.pdf?raw_data=True)
