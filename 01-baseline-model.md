
### 데이터 읽기

 * 참조 kernel : https://www.kaggle.com/rdizzl3/eda-and-baseline-model


```python
import os
for dirname, _, filenames in os.walk('/kaggle/input'):   # 디렉터리명, 파일이름 얻기
    for filename in filenames:
        print(os.path.join(dirname, filename))
```


```python
### 라이브러리 불러오기
import pandas as pd                   # 데이터 처리
import matplotlib.pyplot as plt       # 시각화
import seaborn as sns                 # 시각화
import numpy as np                    # 선형대수 및 행렬 연산

# Import widgets
from ipywidgets import widgets, interactive, interact  # 인터렉티브한 시각화 가능.
import ipywidgets as widgets
from IPython.display import display
```


```python
base_dir = "/kaggle/input/m5-forecasting-accuracy/"  # 기본 경로

submission_file  = pd.read_csv(base_dir+"sample_submission.csv")   # 제출용(60980, 29) - 아이템별 F1~F28 (미래의 향후 28일간 판매량을 예측)
calendar_df  = pd.read_csv(base_dir+"calendar.csv")                # 캘린더(1969,14) - 날짜별 이벤트 및 snap 정보
sellprice = pd.read_csv(base_dir+"sell_prices.csv")                # 판매금액(6841121,4) - 스토아, 아이템별 판매 가격
train_sales  = pd.read_csv(base_dir+"sales_train_validation.csv")  # 판매개수(30490, 1919) - 날짜별(d_1~d_1903) 판매개수
```

### 데이터 간단히 보기


```python
print(calendar_df .columns)
calendar_df[['d','date','event_name_1','event_name_2',
     'event_type_1','event_type_2', 'snap_CA']].head()
```

    Index(['date', 'wm_yr_wk', 'weekday', 'wday', 'month', 'year', 'd',
           'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',
           'snap_CA', 'snap_TX', 'snap_WI'],
          dtype='object')
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>d</th>
      <th>date</th>
      <th>event_name_1</th>
      <th>event_name_2</th>
      <th>event_type_1</th>
      <th>event_type_2</th>
      <th>snap_CA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>d_1</td>
      <td>2011-01-29</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>d_2</td>
      <td>2011-01-30</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>d_3</td>
      <td>2011-01-31</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d_4</td>
      <td>2011-02-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>d_5</td>
      <td>2011-02-02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
## 날짜지정 1~1914 
days = range(1, 1913 + 1)   
time_series_columns = [f'd_{i}' for i in days]

## 30490의 id중에 1000개를 임의로 선택
ids = np.random.choice(train_sales['id'].unique().tolist(), 1000)  

# 위젯 설정.
series_ids = widgets.Dropdown(
    options=ids,
    value=ids[0],
    description='series_ids:'
)

def plot_data(series_ids):
    df = train_sales.loc[train_sales['id'] == series_ids][time_series_columns]  # series_ids의 데이터를 선택
    df = pd.Series(df.values.flatten()) 

    df.plot(figsize=(20, 10), lw=2, marker='*')
    df.rolling(7).mean().plot(figsize=(20, 10), lw=2, marker='o', color='orange') # 이전 7개의 값을 평균을 내어 계산
    plt.axhline(df.mean(), lw=3, color='red')  # 아이템의 평균의 기준 선
    plt.grid()
```

    /opt/conda/lib/python3.6/site-packages/traitlets/traitlets.py:567: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      silent = bool(old_value == new_value)
    


```python
# 아이템을 선택해서 확인
w = interactive(
    plot_data,
    series_ids=series_ids
)
display(w)
```


![png](output_7_0.png)



```python
len(time_series_columns)
```




    1913




```python
# 해당 날짜(d1~d1913)의 행별 값
len(train_sales['d_4'].values)
```




    30490




```python

```


```python
# time_series_columns : 1~1913
# 약 3만개의 데이터 
series_data = train_sales[time_series_columns].values  # train_sales의 d_1 ~ d_1913의 값들

##       1 2 3 4 5 6 7 ----- 1913
##id1    0                            = 50
##id2    0 0 1 1 4 5 2 ---    3       = 25 
##id3000                              = 4
##id3000                              = 5
##id3000         1725                 = 1725
# 판매가 이루어지지 않은날(0개)/1913
pd.Series((series_data == 0).sum(axis=1) / series_data.shape[1]).hist(figsize=(25, 5), color='red')

print(series_data.shape)  # 30490개, 1913 (기간)
print((series_data != 0).argmax(axis=1))  # 각각의 날에 가장 많이 팔린 개수(총 값 30490개)
pd.Series((series_data != 0).argmax(axis=1)).hist(figsize=(25, 5), bins=100) # 30490개의 데이터를 히스토 그램으로 표시
```

    (30490, 1913)
    [ 901  143 1105 ...    1  939 1130]
    




    <matplotlib.axes._subplots.AxesSubplot at 0x7f1afbf37438>




![png](output_11_2.png)



```python
len(series_data == 0)
```




    30490




```python
##       1 2 3 4 5 6 7 ----- 1913
##id1    0                            = 500 / 1913 = 0.25
##id2    0 0 1 1 4 5 2 ---    3       = 200 / 1913 = 0.1
##id3000                              = 1900/ 1913 = 0.9
# 판매가 이루어지지 않은날(0개)/1913
pd.Series((series_data == 0).sum(axis=1) / series_data.shape[1]).hist(figsize=(25, 5), color='red')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f1afdf5e2b0>




![png](output_13_1.png)



```python
# 최대판매개수
print( series_data.max(axis=1) ) # 해당 항목(행)의 최대 판매개수

# 최다 판매개수의 리스트를 출력
pd.Series(series_data.max(axis=1)).value_counts().head(20).plot(kind='bar', figsize=(25, 10))
```

    [ 5  5  6 ... 20 12 12]
    




    <matplotlib.axes._subplots.AxesSubplot at 0x7f1afab34be0>




![png](output_14_2.png)



```python
# 최다 판매개수 꼴지 20개
pd.Series(series_data.max(axis=1)).value_counts().tail(20)
```




    345    1
    601    1
    380    1
    220    1
    204    1
    131    1
    163    1
    211    1
    227    1
    323    1
    355    1
    763    1
    299    1
    634    1
    282    1
    116    1
    132    1
    276    1
    436    1
    367    1
    dtype: int64



### 모델 만들기
 * 예측 값은 가장 최근 28일간의 평균으로 넣어보자.


```python
#    1  2   3  4   5  6  1885 ..... 1913  (28일간의 평균)
# a                      111111111111111 => 1
# b                      00000000000028 28 00 => 2
# c                      222222222222222 => 2

forecast = pd.DataFrame(series_data[:, -28:]).mean(axis=1)  # 28일간의 값의 평균(30490개)
print(forecast.shape)
forecast = pd.concat([forecast] * 28, axis=1)               # 30490개, 28열을 만든다.
print(forecast.shape)
forecast.columns = [f'F{i}' for i in range(1, forecast.shape[1] + 1)] # 컬럼명을 만든다
forecast.head()
```

    (30490,)
    (30490, 28)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>F1</th>
      <th>F2</th>
      <th>F3</th>
      <th>F4</th>
      <th>F5</th>
      <th>F6</th>
      <th>F7</th>
      <th>F8</th>
      <th>F9</th>
      <th>F10</th>
      <th>...</th>
      <th>F19</th>
      <th>F20</th>
      <th>F21</th>
      <th>F22</th>
      <th>F23</th>
      <th>F24</th>
      <th>F25</th>
      <th>F26</th>
      <th>F27</th>
      <th>F28</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.964286</td>
      <td>0.964286</td>
      <td>0.964286</td>
      <td>0.964286</td>
      <td>0.964286</td>
      <td>0.964286</td>
      <td>0.964286</td>
      <td>0.964286</td>
      <td>0.964286</td>
      <td>0.964286</td>
      <td>...</td>
      <td>0.964286</td>
      <td>0.964286</td>
      <td>0.964286</td>
      <td>0.964286</td>
      <td>0.964286</td>
      <td>0.964286</td>
      <td>0.964286</td>
      <td>0.964286</td>
      <td>0.964286</td>
      <td>0.964286</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>...</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.071429</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.571429</td>
      <td>0.571429</td>
      <td>0.571429</td>
      <td>0.571429</td>
      <td>0.571429</td>
      <td>0.571429</td>
      <td>0.571429</td>
      <td>0.571429</td>
      <td>0.571429</td>
      <td>0.571429</td>
      <td>...</td>
      <td>0.571429</td>
      <td>0.571429</td>
      <td>0.571429</td>
      <td>0.571429</td>
      <td>0.571429</td>
      <td>0.571429</td>
      <td>0.571429</td>
      <td>0.571429</td>
      <td>0.571429</td>
      <td>0.571429</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.821429</td>
      <td>1.821429</td>
      <td>1.821429</td>
      <td>1.821429</td>
      <td>1.821429</td>
      <td>1.821429</td>
      <td>1.821429</td>
      <td>1.821429</td>
      <td>1.821429</td>
      <td>1.821429</td>
      <td>...</td>
      <td>1.821429</td>
      <td>1.821429</td>
      <td>1.821429</td>
      <td>1.821429</td>
      <td>1.821429</td>
      <td>1.821429</td>
      <td>1.821429</td>
      <td>1.821429</td>
      <td>1.821429</td>
      <td>1.821429</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.357143</td>
      <td>1.357143</td>
      <td>1.357143</td>
      <td>1.357143</td>
      <td>1.357143</td>
      <td>1.357143</td>
      <td>1.357143</td>
      <td>1.357143</td>
      <td>1.357143</td>
      <td>1.357143</td>
      <td>...</td>
      <td>1.357143</td>
      <td>1.357143</td>
      <td>1.357143</td>
      <td>1.357143</td>
      <td>1.357143</td>
      <td>1.357143</td>
      <td>1.357143</td>
      <td>1.357143</td>
      <td>1.357143</td>
      <td>1.357143</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>



### 예측


```python
validation_ids = train_sales['id'].values
print(validation_ids.shape)
print(validation_ids[0:3])
evaluation_ids = [i.replace('validation', 'evaluation') for i in validation_ids]
```

    (30490,)
    ['HOBBIES_1_001_CA_1_validation' 'HOBBIES_1_002_CA_1_validation'
     'HOBBIES_1_003_CA_1_validation']
    


```python
ids = np.concatenate([validation_ids, evaluation_ids])

predictions = pd.DataFrame(ids, columns=['id'])
forecast = pd.concat([forecast] * 2).reset_index(drop=True)
predictions = pd.concat([predictions, forecast], axis=1)
predictions.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>F1</th>
      <th>F2</th>
      <th>F3</th>
      <th>F4</th>
      <th>F5</th>
      <th>F6</th>
      <th>F7</th>
      <th>F8</th>
      <th>F9</th>
      <th>...</th>
      <th>F19</th>
      <th>F20</th>
      <th>F21</th>
      <th>F22</th>
      <th>F23</th>
      <th>F24</th>
      <th>F25</th>
      <th>F26</th>
      <th>F27</th>
      <th>F28</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>HOBBIES_1_001_CA_1_validation</td>
      <td>0.964286</td>
      <td>0.964286</td>
      <td>0.964286</td>
      <td>0.964286</td>
      <td>0.964286</td>
      <td>0.964286</td>
      <td>0.964286</td>
      <td>0.964286</td>
      <td>0.964286</td>
      <td>...</td>
      <td>0.964286</td>
      <td>0.964286</td>
      <td>0.964286</td>
      <td>0.964286</td>
      <td>0.964286</td>
      <td>0.964286</td>
      <td>0.964286</td>
      <td>0.964286</td>
      <td>0.964286</td>
      <td>0.964286</td>
    </tr>
    <tr>
      <th>1</th>
      <td>HOBBIES_1_002_CA_1_validation</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>...</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.071429</td>
    </tr>
    <tr>
      <th>2</th>
      <td>HOBBIES_1_003_CA_1_validation</td>
      <td>0.571429</td>
      <td>0.571429</td>
      <td>0.571429</td>
      <td>0.571429</td>
      <td>0.571429</td>
      <td>0.571429</td>
      <td>0.571429</td>
      <td>0.571429</td>
      <td>0.571429</td>
      <td>...</td>
      <td>0.571429</td>
      <td>0.571429</td>
      <td>0.571429</td>
      <td>0.571429</td>
      <td>0.571429</td>
      <td>0.571429</td>
      <td>0.571429</td>
      <td>0.571429</td>
      <td>0.571429</td>
      <td>0.571429</td>
    </tr>
    <tr>
      <th>3</th>
      <td>HOBBIES_1_004_CA_1_validation</td>
      <td>1.821429</td>
      <td>1.821429</td>
      <td>1.821429</td>
      <td>1.821429</td>
      <td>1.821429</td>
      <td>1.821429</td>
      <td>1.821429</td>
      <td>1.821429</td>
      <td>1.821429</td>
      <td>...</td>
      <td>1.821429</td>
      <td>1.821429</td>
      <td>1.821429</td>
      <td>1.821429</td>
      <td>1.821429</td>
      <td>1.821429</td>
      <td>1.821429</td>
      <td>1.821429</td>
      <td>1.821429</td>
      <td>1.821429</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HOBBIES_1_005_CA_1_validation</td>
      <td>1.357143</td>
      <td>1.357143</td>
      <td>1.357143</td>
      <td>1.357143</td>
      <td>1.357143</td>
      <td>1.357143</td>
      <td>1.357143</td>
      <td>1.357143</td>
      <td>1.357143</td>
      <td>...</td>
      <td>1.357143</td>
      <td>1.357143</td>
      <td>1.357143</td>
      <td>1.357143</td>
      <td>1.357143</td>
      <td>1.357143</td>
      <td>1.357143</td>
      <td>1.357143</td>
      <td>1.357143</td>
      <td>1.357143</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 29 columns</p>
</div>




```python
# 제출
predictions.to_csv('submission.csv', index=False)
```


```python

```
