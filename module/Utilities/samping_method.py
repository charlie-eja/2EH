import numpy as np
import pandas as pd
from fractions import Fraction


def interval_sampling(data,start_index=0,end_index=-1,interval_count=1,time_index=None):
    '''interval sampling '''
    if interval_count==1:
        interval_data = data.iloc[start_index:end_index:int(interval_count)]
    else:
        score = Fraction(interval_count).limit_denominator(100)
        denominator = score.denominator
        numerator = score.numerator
        if denominator == 1:
            expand_data = data
        else:
            expand_data=expand_timeseries(data, factor=denominator)
        start_index = start_index * denominator if start_index != -1 else start_index
        end_index = end_index * denominator if end_index != -1 else end_index
        interval_data = expand_data.iloc[start_index:end_index:numerator]
    return interval_data

def expand_timeseries(data,factor,time_index='Time') -> pd.DataFrame:
    '''expand data and time  '''
    if time_index is None:
        data = data.reset_index(drop=True)
        old_index = np.arange(len(data))
        new_index = np.linspace(0, len(data) - 1, len(data) * factor)
        data_new = pd.DataFrame(np.empty((len(new_index), len(data.columns))),
                              columns=data.columns)

        for col in data.columns:
            data_new[col] = np.interp(new_index, old_index, data[col])
    else:
        data[time_index] = pd.to_datetime(data[time_index], errors='coerce')
        dt = pd.to_timedelta(detect_time_interval(data[time_index].head(10)))
        data = data.set_index(time_index)
        data = data.sort_index()
        half_dt = dt / factor
        new_index = pd.date_range(start=data.index.min(),
                                  end=data.index.max(),
                                  freq=half_dt)
        data_new = data.reindex(data.index.union(new_index))
        data_new = data_new.interpolate(method='time')
        data_new = data_new.loc[new_index]
        data_new = data_new.reset_index().rename(columns={'index': time_index})
    return data_new

def time_sampling(data,start_time=None,end_time=None,interval_count=None,time_index='Time'):
    ''' time samping '''
    excel_time = pd.to_datetime(data[time_index], errors='coerce')
    if start_time is None:
        start_index=0
    else:
        start_time = pd.to_datetime(start_time)
        idx = excel_time[excel_time > start_time].index
        if len(idx) == 0:
            start_index = None
        else:
            start_index = idx[0]

    if end_time is None:
        end_index=-1
    else:
        end_time = pd.to_datetime(end_time)
        idx = excel_time[excel_time < end_time].index
        if len(idx) == 0:
            end_index = None
        else:
            end_index = idx[-1]+1
    if interval_count is None:
        time_ratio=1
    else:
        '''The Time of the Excel file is time by default'''
        data_time=excel_time.head(10)
        interval_time=detect_time_interval(data_time)[:-1]
        time_ratio=interval_count/int(interval_time)

    interval_data=interval_sampling(data, start_index=start_index, end_index= end_index, interval_count=time_ratio)
    return interval_data



def detect_time_interval(series: pd.Series) -> str:
    """
    Automatically determine the closest interval level based on the time series.
    < 60 seconds: round to the nearest seconds
    ≥ 60 seconds: round to the nearest minute
    """

    times = pd.to_datetime(series)
    deltas = times.diff().dropna().dt.total_seconds()
    if len(deltas) == 0:
        return None

    interval = deltas.round().mode()[0]
    if interval < 60:
        return f"{int(interval)}s"
    else:
        nearest_minute = round(interval / 60)
        return f"{nearest_minute*60}s"



if __name__ == '__main__':
    data=pd.read_excel(r'D:\Pycharm Project\2EH\data\Heat_Recovery_System.xlsx',sheet_name='Sheet2')
    # interval_data=interval_sampling(data,interval_count=2.0)
    interval_data=time_sampling(data,interval_count=1800,start_time='2023-01-02 ',end_time='2023-01-03 22:00:00 ')
    print('end')
