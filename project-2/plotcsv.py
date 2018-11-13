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

def plot_csv_two(traincsv, valcsv,traincsv1, valcsv1):
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

    ax1.set_ylim(0,100)
    ax1.set_title("Accuracy")
    ax1.plot(train_acc,color = "#1f77b4", label='Train,Org')
    ax1.plot(val_acc,color = "darkorange", label='Val,Org')

    ax1.legend(loc='lower right')

    ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4])
    ax2.set_ylim(0,3)
    ax2.set_title("Loss")
    ax2.plot(train_loss, color = "#1f77b4")
    ax2.plot(val_loss, color = "darkorange")


    train_csv1 = pd.read_csv(traincsv1)
    val_csv1 = pd.read_csv(valcsv1)

    train_acc1 = train_csv1.groupby("epoch")["train_prec_avg"].mean().values
    #print(train_csv.groupby("epoch").mean())
    val_acc1 = val_csv1.groupby("epoch")["val_prec_avg"].mean().values
    train_loss1 = train_csv1.groupby("epoch")["train_loss_avg"].mean().values
    val_loss1 = val_csv1.groupby("epoch")["val_loss_avg"].mean().values

    ax1.plot(train_acc1,linestyle='-.',color = "#1f77b4")
    ax1.plot(val_acc1,linestyle='-.' ,color = "darkorange")

    ax2.plot(train_loss1,linestyle='-.', color = "#1f77b4",label='Train,DCF')
    ax2.plot(val_loss1,linestyle='-.', color ="darkorange", label='Val,DCF')
    #ax2.legend(handlelength=3)
    ax2.legend(loc='upper right')
    ax2.set_xlabel("Epoch")



    filename = ".".join( traincsv.split(".")[:-1] ) 
    filename = filename.split("/")[-1]
    filename = "_".join( filename.split("_")[1:] ) + "join.png"
    fig.savefig(filename, dpi=fig.dpi)

    
#plot_csv("./train_vgg.vgg16_bn_256_0.1_0.9_0.003_0.0001.csv","./val_vgg.vgg16_bn_256_0.1_0.9_0.003_0.0001.csv")
plot_csv_two("./train_vgg.vgg16_bn_256_0.1_0.9_0.003_0.0001.csv","./val_vgg.vgg16_bn_256_0.1_0.9_0.003_0.0001.csv","./train_dcfnet.vgg16_bn_256_0.1_0.9_0.003_0.0001.csv","./val_dcfnet.vgg16_bn_256_0.1_0.9_0.003_0.0001.csv")
plot_csv_two("./train_resnet.resnet18_256_0.1_0.9_0.003_0.001.csv","./val_resnet.resnet18_256_0.1_0.9_0.003_0.001.csv","./train_dcfresnet.resnet18_256_0.1_0.9_0.003_0.001.csv","./val_dcfresnet.resnet18_256_0.1_0.9_0.003_0.001.csv")