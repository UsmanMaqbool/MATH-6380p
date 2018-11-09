import pandas as pd
import matplotlib.pyplot as plt
def plot_csv(traincsv, valcsv):
    train_csv = pd.read_csv(traincsv)
    val_csv = pd.read_csv(valcsv)

    train_acc = train_csv.groupby("epoch")["train_prec_avg"].mean().values
    #print(train_csv.groupby("epoch").mean())
    val_acc = val_csv.groupby("epoch")["val_prec_avg"].mean().values
    train_loss = train_csv.groupby("epoch")["train_loss_avg"].mean().values
    val_loss = val_csv.groupby("epoch")["val_loss_avg"].mean().values

    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4],
                   xticklabels=[])
    ax1.set_title("Accuracy")
    ax1.plot(train_acc)
    ax1.plot(val_acc)

    ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4])
    ax2.set_title("Loss")
    ax2.plot(train_loss)
    ax2.plot(val_loss)

    filename = ".".join( traincsv.split(".")[:-1] ) 
    filename = filename.split("/")[-1]
    filename = "_".join( filename.split("_")[1:] ) + ".png"
    fig.savefig(filename, dpi=fig.dpi)


    
plot_csv("./train_vgg.vgg16_bn_256_0.1_0.9_0.003_0.0001.csv","./val_vgg.vgg16_bn_256_0.1_0.9_0.003_0.0001.csv")