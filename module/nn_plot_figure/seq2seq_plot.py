import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_prediction(y,y_pred,output_list,step_index=1):
    index=0
    y = y[:, step_index-1, :]
    y_pred=y_pred[:,step_index-1,:]
    for title in output_list :

        plt.figure(figsize=(6, 5))
        plt.plot(y[:,index],color='b', label='real')
        plt.plot(y_pred[:,index],color='r', label='pred')
        MAE=np.round(np.mean(abs(y[:, index]-y_pred[:,index])),2)
        R2 = np.round(r2_score(y[:, index], y_pred[:,index]),2)
        MAPE = np.round(np.mean(abs((y[:, index] - y_pred[:, index])/y_pred[:, index])), 3)*100
        plt.title(f'{title}  R2 = {R2:.2f}, MAE = {MAE:.2e}, MAPE = {MAPE:.1f}%')
        plt.xlabel("")
        plt.ylabel("Y")
        plt.tight_layout()
        plt.legend()
        plt.savefig(f'figure/{title}_{step_index}.png', dpi=150)
        plt.show()
        index+=1
