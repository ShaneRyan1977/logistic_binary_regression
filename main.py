from typing import List

import pandas as pd
import statsmodels.formula.api as smf
from pandas import Series
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import PolynomialFeatures
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


data_folder_k: str = r"C:\strats\ib_market_data_listener\downloaded_data"
cl_file_name_k: str = "TRADES-CLH4-20230824.csv"
rb_file_name_k: str = "TRADES-RBH4-20230824.csv"
prepped_data_file_name_k: str = data_folder_k + "\\" + "Prepped-Data-BLR.csv"

date_column_k: str = 'Date'
close_column_k: str = 'Close'
volume_column_k: str = 'Volume'
target_column_k: str = 'target'
prefix_k: str = "fac"
perc_ret_k: str = "PRet"


# def titantic_example():
#     titanic = pd.read_csv("titanic.csv")
#     print(titanic.head())
#     # survived,pclass,sex,age,sibsp,parch,fare,embarked,class,who,adult_male,deck,embark_town,alive,alone
#     titanic: pd.DataFrame = titanic[["survived", "adult_male", "sex", "age", "embark_town", "pclass", "alone", "fare"]]
#
#     titanic['pclass'] = titanic['pclass'].astype('category')
#     titanic['sex'] = titanic['sex'].astype('category')
#     titanic['adult_male'] = titanic['adult_male'].astype('bool')
#     print(titanic.head())
#     titanic['alone'] = titanic['alone'].astype('bool')
#     titanic = titanic.dropna()
#     log_reg = smf.logit("survived ~ adult_male + sex + age + embark_town + pclass + alone + fare", data=titanic).fit()
#     print(log_reg.summary())
#     predicted_values1 = log_reg.predict()
#     threshold = 0.5
#     predicted_class1 = np.zeros(predicted_values1.shape)
#     predicted_class1[predicted_values1>threshold] = 1
#     cm1 = confusion_matrix(titanic['survived'], predicted_class1)
#     print('Confusion Matrix : \n', cm1)
#
#     print(classification_report(titanic['survived'], predicted_class1))

class Predictor:

    feat_idx: int = 0

    def next_feat_idx(self):
        feat_str: str = f"{prefix_k}{self.feat_idx}"
        self.feat_idx += 1
        return feat_str

    @staticmethod
    def rolling_avg(data: pd.DataFrame, period: int) -> pd.DataFrame:

        def geo_ret(ser: pd.Series) -> pd.Series:
            return (1 + ser).prod() - 1

        return data[perc_ret_k].rolling(period).apply(geo_ret).shift(1)

    def load_and_prep_data(self, file_name: str) -> pd.DataFrame:
        data = pd.read_csv(data_folder_k + "\\" + file_name)
        data.drop('Open', axis='columns', inplace=True)
        data.drop('High', axis='columns', inplace=True)
        data.drop('Low', axis='columns', inplace=True)
        data.drop('Average', axis='columns', inplace=True)
        data[date_column_k] = pd.to_datetime(data[date_column_k])
        data = data.set_index(date_column_k).sort_index()

        data[perc_ret_k] = data[close_column_k].pct_change()
        data[target_column_k] = data[close_column_k].shift(-10) - data[close_column_k]  # the return in 10 minutes. The negative give you a look ahead
        data['target_2'] = np.where(data[target_column_k] > 0, 1, 0)
        # data['target_2'] = data['target_2'].astype('category')
        data.drop(target_column_k, axis='columns', inplace=True)
        data.rename({'target_2': target_column_k}, axis='columns', inplace=True)

        feat_0_str: str = self.next_feat_idx()
        data[feat_0_str] = data[perc_ret_k].shift(1)  # Previous Min Return
        feat_1_str: str = self.next_feat_idx()
        data[self.next_feat_idx()] = data[perc_ret_k].shift(2)  # Previous Min -1 Return

        # data[self.next_feat_idx()] = data[perc_ret_k].shift(3)  # Previous Min -1 Return
        # data[self.next_feat_idx()] = data[perc_ret_k].shift(4)  # Previous Min -1 Return
        # data[self.next_feat_idx()] = data[perc_ret_k].shift(5)  # Previous Min -1 Return
        # data[self.next_feat_idx()] = data[perc_ret_k].shift(6)  # Previous Min -1 Return
        # data[self.next_feat_idx()] = data[perc_ret_k].shift(7)  # Previous Min -1 Return
        # data[self.next_feat_idx()] = data[perc_ret_k].shift(8)  # Previous Min -1 Return

        # data[self.next_feat_idx()] = (1 + data[feat_0_str])*(1 + data[feat_1_str]) - 1  # Previous two mins Cululative Returns


        # data[self.next_feat_idx()] = self.rolling_avg(data, 5)
        # data[self.next_feat_idx()] = self.rolling_avg(data, 14)
        # data[self.next_feat_idx()] = self.rolling_avg(data, 22)
        # data[self.next_feat_idx()] = self.rolling_avg(data, 120)
        # data[self.next_feat_idx()] = self.rolling_avg(data, 250)

        # one_day_volume_k: str = 'OneDayVolume'
        # data[one_day_volume_k] = data[volume_column_k].rolling(1400).sum().shift(1)
        # data[self.next_feat_idx()] = data[volume_column_k].shift(1) / data[one_day_volume_k]
        # data[self.next_feat_idx()] = data[volume_column_k].rolling(2).sum().shift(1) / data[one_day_volume_k]
        # data[self.next_feat_idx()] = data[volume_column_k].rolling(5).sum().shift(1) / data[one_day_volume_k]
        # data[self.next_feat_idx()] = data[volume_column_k].rolling(14).sum().shift(1) / data[one_day_volume_k]
        # data[self.next_feat_idx()] = data[volume_column_k].rolling(22).sum().shift(1) / data[one_day_volume_k]
        # data[self.next_feat_idx()] = data[volume_column_k].rolling(120).sum().shift(1) / data[one_day_volume_k]

        # data[self.next_feat_idx()] = data[close_column_k].diff()
        #
        # vol_feat_idx: str = self.next_feat_idx()
        # data[vol_feat_idx] = data[volume_column_k].rolling(30).sum()
        # data[self.next_feat_idx()] = data[close_column_k].rolling(5).sum() / 5 - data[close_column_k]
        # data[self.next_feat_idx()] = data[close_column_k].rolling(10).sum() / 10 - data[close_column_k]
        # data[self.next_feat_idx()] = data[close_column_k].rolling(15).sum() / 15 - data[close_column_k]
        # data[self.next_feat_idx()] = data[close_column_k].rolling(30).sum() / 30 - data[close_column_k]
        # data[self.next_feat_idx()] = data[close_column_k].rolling(60).sum() / 60 - data[close_column_k]
        # data[self.next_feat_idx()] = data[volume_column_k] / data[vol_feat_idx]

        data = data.dropna()
        return data

    @staticmethod
    def create_polynomial_permuatations(data: pd.DataFrame) -> pd.DataFrame:
        target_column: Series = data[target_column_k]
        factors: List = Predictor.factors_list(data)
        pf = PolynomialFeatures(degree=3)
        xp = pf.fit_transform(data.drop(target_column_k, axis='columns')[factors])

        factorsp = pf.get_feature_names_out(factors)
        xp = pd.DataFrame(xp, columns=factorsp, index=data.index)
        xp.columns = xp.columns.str.replace(' ', '_')
        xp.columns = xp.columns.str.replace('\t', '_')
        xp.columns = xp.columns.str.replace('^', '_')
        data = pd.concat([target_column, xp], axis='columns').dropna()
        return data

    def regenerate_data(self):

        # rb_data: pd.DataFrame = self.load_and_prep_data("\\" + rb_file_name_k)
        cl_data: pd.DataFrame = self.load_and_prep_data("\\" + cl_file_name_k)
        # rb_data.drop(close_column_k, axis='columns', inplace=True)
        # rb_data.drop(volume_column_k, axis='columns', inplace=True)

        # rb_data.drop(perc_ret_k, axis='columns', inplace=True)
        # rb_data.drop('BarCount', axis='columns', inplace=True)
        # rb_data.drop(target_column_k, axis='columns', inplace=True)
        cl_data.drop(close_column_k, axis='columns', inplace=True)
        cl_data.drop(volume_column_k, axis='columns', inplace=True)
        cl_data.drop('BarCount', axis='columns', inplace=True)

        cl_data.to_csv(prepped_data_file_name_k, sep='\t')
        # df_comb = pd.concat([rb_data, cl_data], axis='columns').dropna()

        # df_comb = Predictor.create_polynomial_permuatations(df_comb)

        # df_comb.to_csv(prepped_data_file_name_k, sep='\t')

    @staticmethod
    def factors_list(data: pd.DataFrame) -> List[str]:
        return data.dropna().filter(like=prefix_k).columns.to_list()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # titantic_example()
    predictor = Predictor()
    predictor.regenerate_data()

    df = pd.read_csv(prepped_data_file_name_k, sep='\t')
    factors_list: [str] = Predictor.factors_list(df)
    factors_str: str = ' + '.join(factors_list)
    params_str: str = f"{target_column_k} ~ {factors_str}"
    log_reg = smf.logit(params_str, data=df).fit()
    print(log_reg.summary())
    predicted_values1 = log_reg.predict()
    threshold = 0.5
    predicted_class1 = np.zeros(predicted_values1.shape)
    predicted_class1[predicted_values1 > threshold] = 1
    cm1 = confusion_matrix(df[target_column_k], predicted_class1)
    print('Confusion Matrix : \n', cm1)

    print(classification_report(df[target_column_k], predicted_class1))
    pass


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
