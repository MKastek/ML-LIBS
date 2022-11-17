# ML-LIBS

My approach to solve [LIBS 2022 quantification contest](https://libs2022.com/). Task was to predict the Cr, Mn, Mo, and Ni content of 15 metal alloys. 
Every sample in dataset contains LIBS spectra:


Two approches of dimension reduction were tested:
- selecting lines,
- PCA analysis.

Final model was built with XGBoost Regressor.

## Final result of the contest
![alt text](https://github.com/MKastek/ML-LIBS/blob/f33801ec87d9302a152d33cdfb89cafd17d85a28/images/final_result_rmse.jpeg?raw_data=True)
