import numpy as np
def normalize_gaussian(data):
    # 預設進來的數據為PD list numpy
    data_np = np.array(data)
    mean_data = np.mean(data_np, axis=0)
    std_data = np.std(data_np, axis=0)
    std_data[std_data==0]=0.01
    normalize_data = (data_np - mean_data) /std_data
    return normalize_data,mean_data,std_data
def main():
    data = [[1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6]]
    normalize_data,mean_data,std_data = normalize_gaussian(data)
    print("原始資料：")
    print(np.array(data))
    print("標準化後：")
    print(normalize_data)
    print("平均")
    print(mean_data)
    print("標準差")
    print(std_data)
    int('s')
if __name__ == "__main__":
    main()