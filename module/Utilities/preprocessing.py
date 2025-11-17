import numpy as np
import pandas as pd
from fractions import Fraction

def interval_sampling(data,start_index=0,end_index=-1,interval_count=1):
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
        data_new = data_new.interpolate(method='time', limit_area="inside")
        data_new = data_new.loc[new_index]
        data_new = data_new.reset_index().rename(columns={'index': time_index})
    return data_new

def time_sampling(data,start_time=None,end_time=None,interval_count=None,time_index='Time',time_low=True):
    ''' time samping '''
    excel_time = pd.to_datetime(data[time_index], errors='coerce')
    if start_time is None:
        start_index=0
    else:
        start_time = pd.to_datetime(start_time)
        idx = excel_time[excel_time >= start_time].index
        if len(idx) == 0:
            start_index = None
        else:
            start_index = idx[0]

    if end_time is None:
        end_index=-1
    else:
        end_time = pd.to_datetime(end_time)
        idx = excel_time[excel_time <= end_time].index
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
    if time_low:
        interval_data=interval_data.iloc[:, 1:].to_numpy(dtype=float)
    else:
        interval_data=interval_data.iloc.to_numpy(dtype=float)
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

def normalize_gaussian(data):
    # default input data  PD or list or numpy   (sample * variables)
    data_np = np.array(data)
    mean_data = np.mean(data_np, axis=0)
    std_data = np.std(data_np, axis=0)
    std_data[std_data==0]=0.01
    normalize_data = (data_np - mean_data) /std_data
    return normalize_data,mean_data,std_data

def mult_interval_sampling():
    print('not yet ')

def mult_time_sampling(data,start_time_list,end_time_list,interval_count=None,time_index='Time',time_low=True):
    if len(start_time_list)==len(end_time_list):
        # interval_data=np.vstack([
        #     time_sampling(
        #         data,
        #         start_time=start_time,
        #         end_time=end_time,
        #         interval_count=interval_count,
        #         time_index=time_index,
        #         time_low=time_low,
        #     )
        #     for start_time, end_time in zip(start_time_list, end_time_list)])
        interval_list=[
            time_sampling(
                data,
                start_time=start_time,
                end_time=end_time,
                interval_count=interval_count,
                time_index=time_index,
                time_low=time_low,
            )
            for start_time, end_time in zip(start_time_list, end_time_list)]
        data_lengths = [arr.shape[0] for arr in interval_list]
        interval_data = np.vstack(interval_list)
        return interval_data,data_lengths
    else:
        print('start_time_list length should be equal to end_time_list length')

def sort_3D_data(data,intput_index=[0,1],out_put_index=[2,3],input_time_step=5,output_time_step=6,jump_step=1):
    data_x = data[:,intput_index]
    _, x_dimension = data_x.shape
    data_y = data[input_time_step:,out_put_index]
    _, y_dimension = data_y.shape
    all_x = np.lib.stride_tricks.sliding_window_view(data_x, (input_time_step, x_dimension)).squeeze(axis=1)
    all_y = np.lib.stride_tricks.sliding_window_view(data_y, (output_time_step, y_dimension)).squeeze(axis=1)
    data_x_3D = all_x[:len(all_y):jump_step]
    data_y_3D = all_y[::jump_step]
    return data_x_3D,data_y_3D

def merge_with_gap(rows, max_gap=1):
    rows = sorted(rows)
    if not rows: return []
    ranges = []
    start = prev = rows[0]
    for r in rows[1:]:
        gap = r - prev - 1
        if gap <= max_gap:
            prev = r
        else:
            ranges.append((start, prev))
            start = prev = r
    ranges.append((start, prev))
    return ranges

def find_nan_data(data,time_index='Time',max_gap=1):
    bad_positions = {}  # {col : [bad_row_index,...]}
    for i, row in data.iterrows():
        for col, val in row.items():
            if col != time_index:
                if not isinstance(val, float):
                    bad_positions.setdefault(col, []).append(i)
                    data.at[i, col] = 0.0
    for col, rows in bad_positions.items():
        for s, e in merge_with_gap(rows, max_gap):
            print(f"{s}~{e} row, col {col} has problems")
    return data

if __name__ == '__main__':
    data=pd.read_excel(r'D:\Pycharm Project\2EH\data\Heat_Recovery_System.xlsx',sheet_name='Sheet2')