import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def WhittakerSmooth(x, w, lambda_, differences=1):
    '''
    Penalized least squares algorithm for background fitting

    input
        x: input data (i.e. chromatogram of spectrum)
        w: binary masks (value of the mask is zero if a point belongs to peaks and one otherwise)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background
        differences: integer indicating the order of the difference of penalties

    output
        the fitted background vector
    '''
    X = np.matrix(x)
    m = X.size
    E = eye(m, format='csc')
    for i in range(differences):
        E = E[1:] - E[:-1]  # numpy.diff() does not work with sparse matrix. This is a workaround.
    W = diags(w, 0, shape=(m, m))
    A = csc_matrix(W + (lambda_ * E.T * E))
    B = csc_matrix(W * X.T)
    background = spsolve(A, B)
    return np.array(background)


def airPLS(x, lambda_=100, porder=1, itermax=15):
    '''
    Adaptive iteratively reweighted penalized least squares for baseline fitting

    input
        x: input data (i.e. chromatogram of spectrum)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting

    output
        the fitted background vector
    '''
    m = x.shape[0]
    w = np.ones(m)
    for i in range(1, itermax + 1):
        z = WhittakerSmooth(x, w, lambda_, porder)
        d = x - z
        dssn = np.abs(d[d < 0].sum())
        if (dssn < 0.001 * (abs(x)).sum() or i == itermax):
            if (i == itermax): print('WARING max iteration reached!')
            break
        w[d >= 0] = 0  # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        w[d < 0] = np.exp(i * np.abs(d[d < 0]) / dssn)
        w[0] = np.exp(i * (d[d < 0]).max() / dssn)
        w[-1] = w[0]
    return z


def read_data(filepath):
    return pd.read_csv(filepath)


def divide_data(data):
    X = data.drop(['Cr','Mn','Mo','Ni'], axis = 1)
    y_Cr = data['Cr']
    y_Mn = data['Mn']
    y_Mo = data['Mo']
    y_Ni = data['Ni']
    y = {'Cr': y_Cr, 'Mn': y_Mn, 'Mo': y_Mo, 'Ni': y_Ni}
    return X, y


def filter_noise(X, treshold):
    X_limited = X.iloc[:,X.columns.astype('float')<220]
    X = X.assign(noise_level = X_limited.apply(max, axis=1))
    return X[X['noise_level'] < treshold].drop(['noise_level'],axis=1)


def correct_baseline(X):
    return X.apply(lambda row : row-airPLS(row), axis = 1)


def limit_wavelength(X,w_min = 225, w_max = 940):
    X_limited = X.loc[:,(X.columns.astype('float')>w_min) & (X.columns.astype('float')<w_max)]
    return X_limited


def update_dependent_variable(X, data_training):
    y_Cr = data_training[data_training.index.isin(X.index)]['Cr']
    y_Mn = data_training[data_training.index.isin(X.index)]['Mn']
    y_Mo = data_training[data_training.index.isin(X.index)]['Mo']
    y_Ni = data_training[data_training.index.isin(X.index)]['Ni']
    return y_Cr, y_Mn, y_Mo, y_Ni


def tune_model(X_train, y_train):
    xgb = XGBRegressor()

    parameters = {'nthread': [4],  # when use hyperthread, xgboost may become slower
                  'objective': ['reg:squarederror'],
                  'learning_rate': [.03, 0.05, .07],  # so called `eta` value
                  'max_depth': [5, 6, 7],
                  'min_child_weight': [4],
                  'subsample': [0.7],
                  'colsample_bytree': [0.7],
                  'n_estimators': [500]}

    xgb_grid = GridSearchCV(xgb,
                            parameters,
                            cv=5,
                            n_jobs=5,
                            verbose=True)

    xgb_grid.fit(X_train, y_train)

    print('Best score:', xgb_grid.best_score_)
    print('Best parameters:', xgb_grid.best_params_)

    return xgb_grid.best_estimator_


def preprocessed_data_pca(X, data_training, element='Mn'):
    X_filtered = filter_noise(X=X, treshold=0.05)
    X_avg_limited = limit_wavelength(X_filtered, w_min=225, w_max=940)
    X_baseline_corrected = correct_baseline(X_avg_limited)
    y_Cr, y_Mn, y_Mo, y_Ni = update_dependent_variable(X_baseline_corrected, data_training)
    y = {'Cr': y_Cr, 'Mn': y_Mn, 'Mo': y_Mo, 'Ni': y_Ni}

    pca = PCA(n_components=15)
    X_normalized = Normalizer(norm='l2').fit_transform(X_baseline_corrected)
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y[element], test_size=0.33, random_state=42)

    pca.fit(X_train)

    projected_train = pca.transform(X_train)

    return projected_train, y_train, pca, X_test, y_test


def plot(x, y, element, r2, mae):
    reg_plot = sns.regplot(x=x, y=y, ci=95);
    reg_plot.set_xlabel('True', fontsize=20)
    reg_plot.set_ylabel('Prediction', fontsize=20)
    reg_plot.set_title(element + r' $R^{2}$: ' + str(r2) + ' MAE: ' + str(mae), fontsize=20)
    reg_plot.grid()
    plt.savefig(os.path.join('report',element,element+'.png'))


def report(y_test, y_pred, element, report_df):
    r2 = np.round(r2_score(y_test, y_pred), 3)
    mae = np.round(mean_absolute_error(y_test, y_pred), 3)
    report_df[element] = [r2, mae]
    report_df.to_csv(os.path.join('report','xgboost_report.csv'))
    return r2, mae


if __name__ == "__main__":

    report_df_pca = pd.DataFrame()
    data_training = read_data(os.path.join('data','train_dataset.csv'))
    X_training, y = divide_data(data_training)

    for element in ['Cr', 'Mn', 'Mo', 'Ni']:
        print(element)

        projected_train, y_train, pca, X_test, y_test = preprocessed_data_pca(X=X_training, data_training=data_training,
                                                                              element=element)
        xgb_tuned_model = tune_model(projected_train, y_train)

        projected_test = pca.transform(X_test)
        y_pred = xgb_tuned_model.predict(projected_test)

        r2, mae = report(y_test, y_pred, element, report_df_pca)
        x = np.array(y_test)
        y = np.array(y_pred)
        plot(x, y, element, r2, mae)