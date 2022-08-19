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
from joblib import dump, load
from sklearn.neural_network import MLPRegressor, MLPClassifier
import pickle as pk


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


def limit_wavelength(X,w_min = 225, w_max = 940):
    X_limited = X.loc[:,(X.columns.astype('float')>w_min) & (X.columns.astype('float')<w_max)]
    return X_limited


def update_dependent_variable(X, data_training):
    y_Cr = data_training[data_training.index.isin(X.index)]['Cr']
    y_Mn = data_training[data_training.index.isin(X.index)]['Mn']
    y_Mo = data_training[data_training.index.isin(X.index)]['Mo']
    y_Ni = data_training[data_training.index.isin(X.index)]['Ni']
    return y_Cr, y_Mn, y_Mo, y_Ni


def tune_xgb_model(X_train, y_train):
    xgb = XGBRegressor(random_state=123)

    parameters = {'nthread': [4],  # when use hyperthread, xgboost may become slower
                  'objective': ['reg:squarederror'],
                  'learning_rate': [.03, 0.05, .07],  # so called `eta` value
                  'max_depth': [5, 6, 7],
                  'min_child_weight': [4],
                  'subsample': [0.7],
                  'colsample_bytree': [0.7, 0.8, 1.0],
                  'n_estimators': [500, 800, 1000]}

    xgb_grid = GridSearchCV(xgb,
                            parameters,
                            cv=10,
                            n_jobs=5,
                            verbose=True)

    xgb_grid.fit(X_train, y_train)

    print('Best score:', xgb_grid.best_score_)
    print('Best parameters:', xgb_grid.best_params_)

    return xgb_grid.best_estimator_


def preprocessed_data_pca(X, data_training, element='Mn',filter=False):
    if filter:
        X = filter_noise(X=X, treshold=0.05)
    X_avg_limited = limit_wavelength(X, w_min=225, w_max=940)
    y_Cr, y_Mn, y_Mo, y_Ni = update_dependent_variable(X_avg_limited, data_training)
    y = {'Cr': y_Cr, 'Mn': y_Mn, 'Mo': y_Mo, 'Ni': y_Ni}

    pca = PCA(n_components=15)
    X_normalized = Normalizer(norm='l2').fit_transform(X_avg_limited)
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y[element], test_size=0.33, random_state=42)

    pca.fit(X_train)

    projected_train = pca.transform(X_train)

    return projected_train, y_train, pca, X_test, y_test


def plot(x, y, element, r2, mae):
    reg_plot = sns.regplot(x=x, y=y, ci=95)
    reg_plot.set_xlabel('True', fontsize=20)
    reg_plot.set_ylabel('Prediction', fontsize=20)
    reg_plot.set_title(element + r' $R^{2}$: ' + str(r2) + ' MAE: ' + str(mae), fontsize=20)
    reg_plot.grid()
    plt.savefig(os.path.join('report',element,element+'.png'))
    plt.clf()


def report(y_test, y_pred, element, report_df):
    r2 = np.round(r2_score(y_test, y_pred), 3)
    mae = np.round(mean_absolute_error(y_test, y_pred), 3)
    report_df[element] = [r2, mae]
    report_df['score'] = ['R2', 'MAE']
    report_df = report_df.set_index('score')
    report_df.to_csv(os.path.join('report','xgboost_report.csv'))
    return r2, mae


def train():
    report_df_pca = pd.DataFrame()
    data_training = read_data(os.path.join('data', 'train_dataset.csv'))
    X_training, y = divide_data(data_training)

    for element in ['Cr', 'Mn', 'Mo', 'Ni']:
        print(element)

        projected_train, y_train, pca, X_test, y_test = preprocessed_data_pca(X=X_training, data_training=data_training,
                                                                              element=element)
        pk.dump(pca, open(os.path.join("saved-pca","pca.pkl"), "wb"))
        xgb_tuned_model = tune_xgb_model(projected_train, y_train)
        dump(xgb_tuned_model, os.path.join('saved-models', element, "xgboost_" + element + ".joblib"))

        projected_test = pca.transform(X_test)
        y_pred = xgb_tuned_model.predict(projected_test)

        r2, mae = report(y_test, y_pred, element, report_df_pca)
        x = np.array(y_test)
        y = np.array(y_pred)
        plot(x, y, element, r2, mae)


def test(pca,element):
    test_df = pd.read_csv(os.path.join('data','test_dataset.csv'))
    targets = test_df['target_name'].unique()

    predictions_df = pd.DataFrame()
    for target in targets:
        predictions = []
        print(target, element)
        target_df = test_df.loc[test_df['target_name'] == target].drop(['target_name'],axis=1)
        target_df_limited = limit_wavelength(target_df)
        target_normalized = Normalizer(norm='l2').fit_transform(target_df_limited)
        target_normalized_df = pd.DataFrame(target_normalized)
        for index, row in target_normalized_df.iterrows():
            xgb_model = load(os.path.join('saved-models',element,'xgboost_'+element+'.joblib'))
            predictions.append(xgb_model.predict(pca.transform(row.values.reshape(1, -1))))

        predictions_df[target] = [np.average(predictions), np.std(predictions)]
    predictions_df['score'] = ['pred', 'unc']
    predictions_df = predictions_df.set_index('score')
    predictions_df.to_csv(os.path.join('test',element,'test_'+element+'.csv'))


def test_all():
    pca = pk.load(open(os.path.join("saved-pca", "pca.pkl"), 'rb'))
    for element in ['Cr', 'Mn', 'Mo', 'Ni']:
        test(pca,element)


if __name__ == "__main__":
    train()
    test_all()


