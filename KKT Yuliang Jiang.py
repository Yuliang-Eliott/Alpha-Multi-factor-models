import pandas as pd
import numpy as np
import talib
import os
import re
import seaborn as sns
import xgboost as xgb
from pandas import DataFrame
import matplotlib.pyplot as plt
from scipy import stats
import math
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
from sklearn.linear_model import Lasso
from keras.models import Sequential
from keras import layers
import keras
from keras.models import Model
from keras.layers import LSTM, Dense, Dropout, Embedding, Input, Masking, GlobalAveragePooling1D
from keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import scipy.optimize as sco



def explore_dataset(file: str) -> dict:
    """
    Load a dataset and perform preliminary exploration.

    Parameters:
    - file (str): The name of the file containing the dataset.

    Returns:
    - dict: A summary of the dataset exploration.
    """
    
    # Load the dataset
    if file.endswith(".zip"):
        df = pd.read_csv(file, compression='zip', header=0, sep=',', quotechar='"')
    else:
        df = pd.read_csv(file)
        
    # Load security reference data
    sec_ref_1 = pd.read_csv('security_reference_data_w_ret1d_1.csv')
    sec_ref_2 = pd.read_csv('security_reference_data_w_ret1d_2.csv')
    
    # Combine the security reference datasets
    sec_ref = pd.concat([sec_ref_1, sec_ref_2], ignore_index=True)

    # Merge the dataset with security reference data based on 'data_date' and 'security_id'
    df = df.merge(sec_ref, on=['data_date', 'security_id'], how='left')
    
    # Convert data_date from int to datetime format
    df['data_date'] = df['data_date'].astype(str)
    df['data_date'] = pd.to_datetime(df['data_date'], format='%Y%m%d')

    # Calculate the time difference between two consecutive non-NA values for each security_id
    def average_difference(series):
        non_na_dates = series.dropna().sort_values()
        return non_na_dates.diff().mean()
    
    avg_diff_per_id = df.groupby('security_id').apply(lambda x: average_difference(x['data_date']))
    
    # Calculate the overall average frequency
    overall_avg_frequency = avg_diff_per_id.mean().days
    
    # Check the length of the data
    data_length = len(df)
    
    # Average number of IDs per day
    avg_ids_per_day = df.groupby('data_date')['security_id'].nunique().mean()
    
    # Count missing values for the value column
    na_count = 100*df.iloc[:, 2].isna().sum()/data_length

    # Get the time span the dataset covers
    time_span_start = df['data_date'].min()
    time_span_end = df['data_date'].max()
    
    # Get the total number of unique security_ids
    unique_ids = df['security_id'].nunique()
    
    # Count the number of tradable securities
    tradable_count = df[df['in_trading_universe'] == 'Y']['security_id'].nunique()

    # Summarize findings
    summary = {
        "File": file,
        "Average Data Frequency (days)": overall_avg_frequency,
        "Data Length": data_length,
        "Avg IDs/Day": avg_ids_per_day,
        "NA Count(%)": na_count,
        "Time Span Start": time_span_start,
        "Time Span End": time_span_end,
        "Total Unique IDs": unique_ids,
        "Tradable Unique IDs": tradable_count
    }
    
    return summary




files = [x for x in os.listdir() if 'data_set' in x]
files_sorted = sorted(files, key=lambda x: int(re.search(r'data_set_(\d+)', x).group(1)))
summaries = [explore_dataset(file) for file in files_sorted]
summary_df = pd.DataFrame(summaries)




def merge_datasets(files: list) -> pd.DataFrame:
    """
    Merge multiple datasets into one dataframe based on 'data_date' and 'security_id'.
    
    Parameters:
    - files (list): List of file names to merge.
    
    Returns:
    - pd.DataFrame: A merged dataframe.
    """
    dfs = []

    # Sort files by the dataset number
    files_sorted = sorted(files, key=lambda x: int(re.search(r'data_set_(\d+)', x).group(1)))

    # Load and rename columns
    for file in files_sorted:
        # Extract the dataset number from the file name
        num = file.split('_')[2].split('.')[0]

        # For zip files, use the `compression` parameter
        compression = 'zip' if file.endswith('.zip') else None

        # Load the dataset
        df = pd.read_csv(file, compression=compression)
        df['data_date'] = pd.to_datetime(df['data_date'].apply(str))
        # Solve the question the same day and smae id has two different values, we take the mean
        df = df.groupby(['data_date', 'security_id']).mean().drop_duplicates()
        dfs.append(df)

    # Merge all the dataframes
    merged_df = pd.concat(dfs,axis=1,join = 'outer').reset_index()
    ## First fill the NA by the forward value of the same id
    merged_df = merged_df.groupby('security_id').apply(lambda group: group.sort_values(by='data_date').ffill()).reset_index(drop=True)
    ## Second fill the NA by the mean value of the same date
    merged_df = merged_df.groupby('data_date').apply(lambda group: group.fillna(group.mean())).reset_index(drop=True)

    sec_ref_1 = pd.read_csv('security_reference_data_w_ret1d_1.csv')
    sec_ref_2 = pd.read_csv('security_reference_data_w_ret1d_2.csv')
    sec_ref = pd.concat([sec_ref_1,sec_ref_2])
    
    ## Clear out the extrme return that greater than 1
    sec_ref = sec_ref[sec_ref['ret1d']<=1]
    
    ## Calculate Excess Return 
    def handle(x):
        x['excess_ret1d'] = x['ret1d'] - x['ret1d'].mean()
        return x.set_index('security_id')[x.columns.difference(['data_date','security_id'])]
    sec_ref = sec_ref.groupby('data_date').apply(lambda x:handle(x)).reset_index()
    sec_ref['data_date'] = pd.to_datetime(sec_ref['data_date'].apply(str))
    
    merged_df = pd.merge(merged_df,sec_ref,on=['data_date', 'security_id'],how='left')
    
    return merged_df.dropna()




files = [x for x in os.listdir() if 'data_set' in x]
merged_df = merge_datasets(files)



def compute_factors(data):
    # Sorting the dataframe by security_id and date
    data = data.sort_values(by=['security_id', 'data_date'])

    # List to store dataframes after processing each security's data
    dfs = []

    for security, df in data.groupby('security_id'):
        
        df = df.sort_values(by='data_date')
        
        # 1. Moving Average
        for i in range(6,51,4):
            df[f'SMA_{i}'] = talib.SMA(df['close_price'], timeperiod=i)

        # 2. EMA - Exponential Moving Average
        for i in range(6,51,4):
            df[f'EMA_{i}'] = talib.EMA(df['close_price'], timeperiod=i)

        # 3. VWMA - Volume Weighted Moving Average
        vol_price = df['volume'] * df['close_price']
        for i in range(6,51,4):
            df[f'VSMA_{i}'] = talib.SMA(vol_price, timeperiod=i)

        # 4. BBANDS - Bollinger Bands with different timeperiod
        for i in range(14,61,6):
            upper, middle, lower = talib.BBANDS(df['close_price'], timeperiod=i)
            df[f'BBANDS_upper_{i}'] = upper
            df[f'BBANDS_middle_{i}'] = middle
            df[f'BBANDS_lower_{i}'] = lower

        # 5. MOM - Momentum
        for i in range(14,61,6):
            df[f'MOM_{i}'] = talib.MOM(df['close_price'], timeperiod=i)


        # 6. Acceleration - Difference in the change of momentum
        for i in range(14,61,6):
            df[f'ACCEL_{i}'] = df[f'MOM_{i}'].diff()


        # 7. Rate of Change - Rate change of Price
        for i in range(14,61,6):
            df[f'ROCR_{i}'] = df['close_price'].pct_change(i)

        # 6. Moving Average Convergence Divergence
        for i in [18,24,30]:
            macd, macdsignal, macdhist = talib.MACD(df['close_price'], fastperiod=12, slowperiod=i)
            df[f'MACD_12_{i}'] = macd

        # 7. RSI
        for i in [8,14,20]:
            df[f'RSI_{i}'] = talib.RSI(df['close_price'], timeperiod=i)

        # 8. Price Volume Trend
        df['PVT'] = df['volume'] * (df['close_price'].pct_change())

        # 9. OBV - On Balance Volume
        df['OBV'] = talib.OBV(df['close_price'], df['volume'])

        # 10.Psychological Line Indicator
        df['PSY'] =(df['close_price'] > df['close_price'].shift(1)).rolling(window=14).sum() / 14 * 100
        
        
        #11.Create volatility by rolling 3, 5, 15 days
        ret = df['close_price'].pct_change()
        for i in [3,5,15]:
            df[f'sd_{i}'] = ret.rolling(window = i).std()

        df['sd5_15'] = df['sd_5']/df['sd_15']
    
        #12. Create 5, 15 days moving volume volatility
        for i in [3,5,15]:
            df[f'volsd_{i}'] = df['volume'].rolling(window = i).std()
            
        df['volsd5_15'] = df['volsd_5']/df['volsd_15']
    
        #13. Correlation between 
        df['vol_change'] = df['volume'].pct_change()
        for i in [5,15]:
            df[f'corr_{i}'] = ret.rolling(i).corr(df['vol_change'])

        ## Predict return label(futurn 1day excess return)
        df['target'] = df['excess_ret1d'].shift(-1)
        df['tmr_ret1d'] = df['ret1d'].shift(-1)
        
        
        # Append the processed data to our list
        dfs.append(df)


    # Concatenate all the processed dataframes together
    final_df = pd.concat(dfs, ignore_index=True)
    
    return final_df.dropna()



all_df = compute_factors(merged_df)
all_df = all_df.set_index(['data_date','security_id']).sort_index()




class AlphaSignalAnalyzer:
    def __init__(self, alpha_signal_df: DataFrame, factor_name: str, price_data: DataFrame):
        self.factor_df = alpha_signal_df
        self.factor_df.index.names = ['date','id']
        self.factor_name = factor_name
        
        self.corr_method = 'pearson'
        self.k_layers = 10  # Adjust as needed
        self.portfolio_stock_num = 10  # Adjust as needed
        
        self.return_type = ['return_1','return_2','return_5']
        self.price_data = price_data
        self.price_data.index.names = ['date','id']
        self.price_data = self.price_data.reset_index()

        self.layered_ret_dfs = dict()
        self.ls_ret_dfs = dict()

    def run(self):
        print("-" * 50)
        print("Running analysis...")
        self._add_returns()
        self._calc_sdav_ic()
        for layered_ret_type in self.return_type:
            self._calc_layered_ret(layered_ret_type)
        self._backtest_top_stocks()
        self._gen_report()

    def _add_returns(self):
        print("Adding returns...")
        for i in self.return_type:
            return_data = self.price_data.groupby('id').apply(lambda x:x.set_index('date')['close_price'].pct_change(int(i[-1])).shift(-int(i[-1]))).to_frame(i)
            return_data = return_data[return_data[i]<=1]
            self.factor_df = pd.merge(self.factor_df.reset_index(),return_data.reset_index(),on = ['date','id'],how='left').dropna()
            ## Calculate excess return (demean)
            def handle(x,return_type):
                x[return_type] = x[return_type] - x[return_type].mean()
                return x
            self.factor_df = self.factor_df.groupby('date').apply(lambda x:handle(x,i)).reset_index(drop=True)
            
        self.factor_df = self.factor_df.set_index(['date','id'])
        
        
        
    def _calc_layered_ret(self, layered_ret_type):
        print(f"Calculating layered return of {layered_ret_type}...")
        layered_ret_df = self.factor_df[[self.factor_name, layered_ret_type]].copy().reset_index()
        
        layered_ret_df['layer'] = (layered_ret_df.groupby("date")[self.factor_name].rank(
            pct=True, method='first') * self.k_layers).astype(int) + 1
        layered_ret_df.loc[layered_ret_df.layer > self.k_layers, 'layer'] = self.k_layers
        layered_ret_df = (layered_ret_df.groupby(['date', 'layer'])[layered_ret_type]
                          .mean().unstack().cumsum())

        self.layered_ret_dfs[layered_ret_type] = (layered_ret_df.stack().reset_index()
                                                  .rename(columns={0: layered_ret_type}))
        
        self.ls_ret_dfs[layered_ret_type] = pd.DataFrame({
            self.k_layers // 2 - layer + 1: layered_ret_df[self.k_layers - layer + 1] - layered_ret_df[layer]
            for layer in range(1, self.k_layers // 2 + 1)
        }).stack().reset_index().rename(columns={'level_1': 'layer', 0: layered_ret_type})

    def _calc_sdav_ic(self):
        print("Calculating IC & IR...")
        self.ic_df = (self.factor_df[[self.factor_name] + self.return_type].reset_index().groupby("date")
                      .apply(lambda x: x.corr(method=self.corr_method)[self.factor_name])
                      .drop(self.factor_name, axis=1).stack().reset_index()
                      .rename(columns={self.factor_name: "Type", 0: "IC"}))
        
        self.ic_df = self.ic_df[self.ic_df['Type']!='id']
        
        self.ic_df['year'] = self.ic_df['date'].apply(lambda x: x.year)

        self.ir_df = (self.ic_df.groupby(['year', 'Type']).apply(lambda x: x['IC'].mean() / x['IC'].std())
                      .reset_index().rename(columns={0: "IR"}))

    def _backtest_top_stocks(self):
        print("Backtesting top stocks...")
        self.port_df = self.factor_df.copy().reset_index()
        self.port_df['rank'] = self.port_df.groupby("date")[self.factor_name].rank(method='first', ascending=False)
        self.port_df = self.port_df[
            self.port_df['rank'] <= self.portfolio_stock_num].sort_values(by=['date', 'rank'])
        self.port_df['rank'] = self.port_df['rank'].astype(str)

        res_list = list()
        weights_df = pd.pivot(data=self.port_df, index='date', columns='rank', values=self.factor_name)
        weights_df = weights_df.div(weights_df.sum(axis=1), axis=0)
        for portfolio_ret_type in self.return_type:
            ret_df = pd.pivot(data=self.port_df, index='date', columns='rank', values=portfolio_ret_type)
            res_list.append(ret_df.mul(weights_df).sum(axis=1).to_frame(portfolio_ret_type))
        port_ret_df = pd.concat(res_list, axis=1)
        port_cum_ret_df = port_ret_df.cumsum().rename(columns={col: f"cum_{col}" for col in port_ret_df.columns})
        self.port_ret_df = (pd.concat([port_ret_df, port_cum_ret_df], axis=1).stack()
                            .reset_index().rename(columns={'level_1': 'Type', 0: 'Returns'}))

        stocks_df = pd.pivot(data=self.port_df, index='date', columns='rank', values='id').reset_index()

    def _gen_report(self):
        print("Generating report...")
        fig_num = len(self.return_type)*2+3
        fig = plt.figure(figsize=(25, 20))
        axes = fig.subplots(nrows=math.ceil(fig_num/3), ncols=3).flatten()
        counter = 0

        for layered_ret_type in self.return_type:
            sns.set_context("paper")
            sns.lineplot(x='date', y=layered_ret_type, hue='layer', palette="RdYlGn_r",
                         data=self.layered_ret_dfs[layered_ret_type], ax=axes[counter])
            axes[counter].set_title(f"[{self.factor_name}] Hedged Layered Ret")
            axes[counter].grid()
            counter += 1

        for layered_ret_type in self.return_type:
            sns.set_context("paper")
            sns.lineplot(x='date', y=layered_ret_type, hue='layer', palette="RdYlGn_r",
                         data=self.ls_ret_dfs[layered_ret_type], ax=axes[counter])
            axes[counter].set_title(f"[{self.factor_name}] Long Short Ret")
            axes[counter].grid()
            counter += 1

        sns.set_context("paper")
        sns.lineplot(x='date', y="IC", hue='Type', palette='Set2', data=self.ic_df, ax=axes[counter])
        axes[counter].set_title(f"[{self.factor_name}] Information Coefficient")
        axes[counter].grid()
        counter += 1
        

        sns.set_context("paper")
        sns.barplot(x='year', y="IR", hue='Type', palette='Set2', data=self.ir_df, ax=axes[counter])
        axes[counter].set_title(f"[{self.factor_name}] Information Ratio")
        axes[counter].grid()
        counter += 1

        sns.set_context("paper")
        sns.lineplot(x='date', y="Returns", hue='Type', palette='RdYlGn_r', data=self.port_ret_df, ax=axes[counter])
        axes[counter].set_title(f"[{self.factor_name}|k={self.portfolio_stock_num}] Top Stocks Return")
        axes[counter].grid()

        plt.tight_layout()
        plt.show()




train_edate = pd.to_datetime("20151231")
valid_edate = pd.to_datetime("20161231")
df_train = all_df.loc[:train_edate]
df_valid = all_df.loc[train_edate:valid_edate]
df_test = all_df.loc[valid_edate:]




df_train_x = df_train[df_train.columns.difference(['close_price','excess_ret1d','group_id','in_trading_universe',
                                                   'ret1d','volume','target'])]
df_train_y = df_train[['target']]

df_valid_x = df_valid[df_valid.columns.difference(['close_price','excess_ret1d','group_id','in_trading_universe',
                                                   'ret1d','volume','target'])]
df_valid_y = df_valid[['target']]

df_test_x = df_test[df_test.columns.difference(['close_price','excess_ret1d','group_id','in_trading_universe',
                                                   'ret1d','volume','target'])]
df_test_y = df_test[['target']]


# In[23]:


sigma = df_train_x.groupby('security_id').std()
mu = df_train_x.groupby('security_id').mean()

df_train_x = df_train_x.groupby('security_id').apply(lambda x:((x-mu)/sigma)).droplevel(2).swaplevel(0,1).sort_index().replace([np.inf, -np.inf], np.nan).dropna()
df_test_x = df_test_x.groupby('security_id').apply(lambda x:((x-mu)/sigma)).droplevel(2).swaplevel(0,1).sort_index().replace([np.inf, -np.inf], np.nan).dropna()
df_valid_x = df_valid_x.groupby('security_id').apply(lambda x:((x-mu)/sigma)).droplevel(2).swaplevel(0,1).sort_index().replace([np.inf, -np.inf], np.nan).dropna()

df_train_y = df_train_y.loc[df_train_x.index]
df_test_y = df_test_y.loc[df_test_x.index]
df_valid_y = df_valid_y.loc[df_valid_x.index]


# In[25]:


# Usage
analyzer = AlphaSignalAnalyzer(df_train_x[['d11']], factor_name="d11", price_data = df_train[['close_price']])
analyzer.run()


# In[27]:


analyzer = AlphaSignalAnalyzer(-df_train_x[['d10']], factor_name="d10", price_data = df_train[['close_price']])
analyzer.run()


# ## XGBoost

# In[28]:


seed = 2023
model_xgb_params= {
    'max_depth':3,
    'nthread':8,
    'objective':'reg:squarederror',
    'eta':0.025,
    'seed':seed,
 }

def pearson_ic(d_pred, d_train):
    d_label = d_train.get_label()
    ic = np.corrcoef(d_label,d_pred)[0][1]
    return 'ic', ic


# In[29]:


dtrain = xgb.DMatrix(df_train_x.values, label=df_train_y.values)
dvalid = xgb.DMatrix(df_valid_x.values, label=df_valid_y.values)


# In[31]:


model_xgb = xgb.train(
    model_xgb_params,
    dtrain,
    400,
    [(dtrain, 'dtrain'),(dvalid,'dvalid')],
    verbose_eval=20,
    feval=pearson_ic
)


# In[32]:


dtest = xgb.DMatrix(df_test_x)
se_pred=model_xgb.predict(dtest)
np.corrcoef(se_pred, df_test_y.values.flatten())


# In[33]:


result_xgb = pd.DataFrame(model_xgb.predict(xgb.DMatrix(df_train_x)), index=df_train_y.index, columns=["xgb_predict"])


# In[34]:


np.corrcoef(result_xgb.values.flatten(), df_train_y.values.flatten())


# In[35]:


result_xgb


# In[36]:


num_features = 10

# Get feature importance
f_importance_dict = model_xgb.get_score(importance_type='weight')

# Sort features based on importance
sorted_features = sorted(f_importance_dict.items(), key=lambda x: x[1], reverse=True)

# Select only the top features
top_features = dict(sorted_features[:num_features])

# Plot the importance for top features
xgb.plot_importance(top_features)
plt.show()


# ## XGBoost Performance in Train Set

# In[37]:


analyzer = AlphaSignalAnalyzer(result_xgb, factor_name="xgb_predict", price_data = df_train[['close_price']])
analyzer.run()


# ## XGBoost Performance in test set

# In[39]:


analyzer = AlphaSignalAnalyzer(pd.DataFrame(se_pred, index=df_test_y.index, columns=["xgb_predict"]), factor_name="xgb_predict", price_data = df_test[['close_price']])
analyzer.run()


# In[40]:


model = LinearRegression()
model.fit(pd.concat([df_train_x,df_valid_x]),pd.concat([df_train_y,df_valid_y]))


# In[41]:


a  = model.intercept_
b = model.coef_


# ## Linear Regression Performance in test set

# In[42]:


analyzer = AlphaSignalAnalyzer(pd.DataFrame(model.predict(df_test_x), index=df_test_y.index, columns=["lr_predict"]), factor_name="lr_predict", price_data = df_test[['close_price']])
analyzer.run()


# In[67]:


lasso = Lasso(alpha=2e-4, max_iter=10000)
lasso.fit(pd.concat([df_train_x,df_valid_x]),pd.concat([df_train_y,df_valid_y]))


# In[68]:


lasso_predict = lasso.predict(df_test_x)


# In[69]:


lasso_result = pd.DataFrame(data=lasso.coef_.tolist(),index=df_train_x.columns.tolist(),columns=['coef']).sort_values(by='coef', ascending=False)


# In[70]:


lasso_result[lasso_result['coef']!=0]


# In[71]:


analyzer = AlphaSignalAnalyzer(pd.DataFrame(lasso_predict, index=df_test_y.index, columns=["lasso_predict"]), factor_name="lasso_predict", price_data = df_test[['close_price']])
analyzer.run()


# In[73]:


selected_features = list(set(list(top_features.keys())) | set(lasso_result[lasso_result['coef']!=0].index))
selected_features


# In[63]:


dtrain = xgb.DMatrix(pd.concat([df_train_x,df_valid_x]).values, label=pd.concat([df_train_y,df_valid_y]).values)
model_xgb = xgb.train(
    model_xgb_params,
    dtrain,
    300,
    [(dtrain, 'dtrain')],
    verbose_eval=20,
    feval=pearson_ic
)
dtest = xgb.DMatrix(df_test_x)
se_pred=model_xgb.predict(dtest)
np.corrcoef(se_pred, df_test_y.values.flatten())


# In[64]:


analyzer = AlphaSignalAnalyzer(pd.DataFrame(se_pred, index=df_test_y.index, columns=["xgb_predict"]), factor_name="xgb_predict", price_data = df_test[['close_price']])
analyzer.run()


# In[82]:


model = Sequential()
model.add(layers.Dense(128, activation='relu',input_dim=(29)))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))
    
# Change an optimizers' learning rate
adam = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=adam, loss='mean_squared_error')
    
# Train 10 times
history = model.fit(df_train_x[selected_features], df_train_y, 
          epochs=10, batch_size=256,validation_data=(df_valid_x[selected_features], df_valid_y), shuffle=False)


# In[83]:


true = df_test_y
predict = pd.DataFrame(model.predict(df_test_x[selected_features]), 
                       index=df_test_x.index, columns=['DNN_predict'])
df_pre = pd.concat([true,predict], axis=1)


# In[84]:


# Make pictures to observe result
fig1 = plt.figure(figsize=(10,6))
plt.plot(history.history['loss'], color='coral', label='loss')
plt.plot(history.history['val_loss'], color='royalblue', label='val_loss')
plt.legend()
plt.grid(True)


# In[ ]:


analyzer = AlphaSignalAnalyzer(predict, factor_name="DNN_predict", price_data = df_test[['close_price']])
analyzer.run()


# In[91]:


x_train = np.reshape(df_train_x, (df_train_x.shape[0], df_train_x.shape[1], 1))
y_train = np.array(df_train_y)

x_valid = np.reshape(df_valid_x, (df_valid_x.shape[0], df_valid_x.shape[1], 1))
y_valid = np.array(df_valid_y)


def LSTM_model():
    
    model = Sequential()
    
    model.add(LSTM(units = 100, return_sequences = True, input_shape = (x_train.shape[1],1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 100))
    model.add(Dropout(0.2))
    
    model.add(Dense(units=1))
    
    return model

model = LSTM_model()
model.summary()
model.compile(optimizer='adam', 
              loss='mean_squared_error')

checkpointer = ModelCheckpoint(filepath = 'weights_best.hdf5', 
                               verbose = 2, 
                               save_best_only = True)

model.fit(x_train, 
          y_train, 
          epochs=10, 
          batch_size = 256,
          validation_data=(x_valid, y_valid),
          callbacks = [checkpointer])


# In[92]:


y_test = np.array(df_test_y)
x_test = np.reshape(df_test_x, (df_test_x.shape[0], df_test_x.shape[1] ,1))

price_pred_lstm = model.predict(x_test)


# In[94]:


lstm_predict = pd.DataFrame(price_pred_lstm, index=df_test_x.index, columns=['LSTM_predict'])


# In[95]:


analyzer = AlphaSignalAnalyzer(lstm_predict, factor_name="LSTM_predict", price_data = df_test[['close_price']])
analyzer.run()


# In[64]:


model = Sequential()
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))

optimizer = Adam(lr=0.0001)
model.compile(loss='mean_squared_error', optimizer=optimizer)

X_train_LSTM ,Y_train_LSTM = convert_data_shape(df_train_x,df_train_y)
X_valid_LSTM ,Y_valid_LSTM = convert_data_shape(df_valid_x,df_valid_y)

history = model.fit(X_train_LSTM ,Y_train_LSTM, 
          epochs=10, validation_data=(X_valid_LSTM ,Y_valid_LSTM),shuffle=False)


# In[267]:


class PortfolioManager:
    def __init__(self, predictions,history, all_df, trading_cost_rate=0.0001, top_n=10):
        self.predictions = predictions
        self.predictions.index.names = ['date','id']
        self.history = history
        self.history.index.names = ['date','id']
        self.all_df = all_df
        self.trading_cost_rate = trading_cost_rate
        self.top_n = top_n
        self.portfolio_value = {'Portfolio': [100000000]}
        self.current_positions = pd.Series(index=self.predictions.columns)
        self.turnovers = []
        self.positions = pd.DataFrame(index = predictions.index.levels[0],columns=['long','short'])
        self.long_returns = []
        self.short_returns = []
        
    def Portfolio_volatility(self, weights, mean_returns,cov_matrix):
        """
        Calculate the expected portfolio volatility.
        """
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def determine_weights(self, returns):
        """
        Determine the optimal weights for the portfolio assets using the Mean-Variance Optimization.
        """
        mean_returns = returns.mean()
        cov_matrix = returns.cov()

        num_assets = len(mean_returns)

        args = (mean_returns, cov_matrix)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bound = (0, 0.1)  # Set the maximum weight for any asset
        bounds = tuple(bound for asset in range(num_assets))

        result = sco.minimize(self.Portfolio_volatility, num_assets * [1. / num_assets], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
        weights = result['x']
        return weights

    def _update_turnover(self, new_positions):
        if self.current_positions.dropna().empty:
            turnover = 0
        else:
            turnover = (self.current_positions.fillna(0) - new_positions.fillna(0)).abs().sum() / 2
        self.turnovers.append(turnover)

    def calculate_portfolio(self):
        
        for date, preds in self.predictions.groupby('date'):
            # Filter stocks in the trading universe
            preds = preds.droplevel(0)
            tradable_stocks = self.all_df[self.all_df['in_trading_universe'] == 'Y'].loc[date].index
            tradable_stocks = list(set(tradable_stocks) & set(preds.index))
            if len(tradable_stocks)<2*self.top_n:
                top_n = len(tradable_stocks)//2
            else:
                top_n = self.top_n

            # Get top and bottom stocks
            long_stocks = preds.loc[tradable_stocks].nlargest(top_n,columns=preds.columns).index.tolist()
            short_stocks = preds.loc[tradable_stocks].nsmallest(top_n,columns=preds.columns).index.tolist()
            
            long_his_returns = self.history.swaplevel().loc[long_stocks].unstack().T.droplevel(0)
            short_his_returns = self.history.swaplevel().loc[short_stocks].unstack().T.droplevel(0)
            
            weights_long = self.determine_weights(long_his_returns)
            weights_short = self.determine_weights(short_his_returns)

            # Determine position size per stock
            position_size = self.portfolio_value['Portfolio'][-1] / 2  # Divided by 2 because we're allocating half to longs and half to shorts

            # Get returns for these stocks the next day
            long_returns = self.all_df.loc[date].loc[long_stocks]['tmr_ret1d']
            short_returns = self.all_df.loc[date].loc[short_stocks]['tmr_ret1d']
            
            long_price = self.all_df.loc[date].loc[long_stocks]['close_price']
            short_price = self.all_df.loc[date].loc[short_stocks]['close_price']

            # Calculate portfolio return and update value
            daily_return = ((long_returns*weights_long).sum() - (short_returns*weights_short).sum()) / 2  # Average return of longs minus shorts
            self.long_returns.append((long_returns*weights_long).sum())
            self.short_returns.append((short_returns*weights_short).sum())
            
            # Update turnover and current positions
            new_positions = pd.Series(index= preds.index)
            new_positions[long_stocks] = position_size / sum(weights_long*long_price)
            new_positions[short_stocks] = -position_size / sum(weights_short*short_price)
            
            self._update_turnover(new_positions)
            trading_cost = self.turnovers[-1] * self.trading_cost_rate  # Cost for both buying and selling
            daily_return -= trading_cost/self.portfolio_value['Portfolio'][-1]
            self.portfolio_value['Portfolio'].append(self.portfolio_value['Portfolio'][-1] * (1 + daily_return))

            
            self.positions.loc[date]['long'] = long_stocks
            self.positions.loc[date]['short'] = short_stocks
            self.current_positions = new_positions
                
    def calculate_sharpe_ratio(self):
        returns = pd.Series(self.portfolio_value['Portfolio']).pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std()
        return sharpe_ratio

    def plot_result(self):
        plt.figure(figsize=(12, 8))
        
        # Portfolio Value
        plt.subplot(2, 2, 1)
        plt.plot(self.portfolio_value['Portfolio'], label="Portfolio Value")
        plt.title('PnL Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.grid(True)
        
        # Portfolio Cumulative Return
        plt.subplot(2, 2, 2)
        returns = pd.Series(self.portfolio_value['Portfolio']).pct_change().cumsum()
        plt.plot(returns, label="Cumulative Returns")
        plt.title('Cumulative Returns over Time')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return (%)')
        plt.legend()
        plt.grid(True)
        
        # Turnover
        plt.subplot(2, 2, 3)
        plt.plot(self.turnovers)
        plt.title('Portfolio Turnover over Time')
        plt.xlabel('Date')
        plt.ylabel('Turnover')
        plt.grid(True)
        
        
        # Long Short Return
        plt.subplot(2, 2, 4)
        plt.plot(np.cumprod(1 + np.array(self.long_returns)), label='Long Cumulative Return')
        plt.plot(np.cumprod(1 + np.array(self.short_returns)), label='Short Cumulative Return')
        plt.xlabel('Date')
        plt.ylabel('Return')
        plt.title('Long and Short Cumulative Return')
        plt.legend()
        plt.grid(True)
        
        
        plt.tight_layout()
        plt.show()

        
    def annualized_return(self):
        total_return = self.portfolio_value['Portfolio'][-1] / self.portfolio_value['Portfolio'][0] - 1
        years = len(self.portfolio_value['Portfolio']) / 252  # Assuming 252 trading days in a year
        annualized_return = (1 + total_return) ** (1/years) - 1
        return annualized_return

    def max_drawdown(self):
        returns = np.array(self.portfolio_value['Portfolio'])
        running_max = pd.Series(returns).cummax().values
        drawdowns = (running_max - returns) / running_max
        return drawdowns.max()

    def position_overview(self):
        long_positions = len(self.current_positions[self.current_positions == 1])
        short_positions = len(self.current_positions[self.current_positions == -1])
        
        print(f"Long Positions: {long_positions}")
        print(f"Short Positions: {short_positions}")

    def summary(self):
        print("Portfolio Summary")
        print("------------------")
        print(f"Sharpe Ratio: {self.calculate_sharpe_ratio():.3f}")
        print(f"Annualized Return: {self.annualized_return():.3f}")
        print(f"Maximum Drawdown: {self.max_drawdown():.3f}")
        self.position_overview()


# In[268]:


portfolio_manager = PortfolioManager(pd.DataFrame(lasso_predict, index=df_test_y.index, columns=["lasso_predict"]),df_train_y, all_df)
portfolio_manager.calculate_portfolio()
portfolio_manager.plot_result()
portfolio_manager.summary()


# In[269]:


portfolio_manager.positions.dropna()


# In[271]:


portfolio_manager = PortfolioManager(pd.DataFrame(se_pred, index=df_test_y.index, columns=["xgb_predict"]),df_train_y, all_df)
portfolio_manager.calculate_portfolio()
portfolio_manager.plot_result()
portfolio_manager.summary()


# In[272]:


portfolio_manager.positions.dropna()


# In[ ]:




