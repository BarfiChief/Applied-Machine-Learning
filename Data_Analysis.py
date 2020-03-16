import pandas as pd
import matplotlib.pyplot as plt
def plot_data(t,x_value,y_value,z_value,y_label,title):
    plt.figure(figsize=(20,10))
    plt.scatter(t,x_value,s=.2,c='r')
    plt.scatter(t,y_value,s=.2,c='g')
    plt.scatter(t,z_value,s=.2,c='b')
    plt.xlabel('Time')
    plt.ylabel(y_label)
    plt.title(title)
    plt.show();
activities=["climbingdown","climbingup","jumping","lying","running","sitting","standing","walking"]
for sub in range(1,7):
    for act in activities:
        file_acc="./Raw_Sensor_Data"+"/Subject_"+str(sub)+"/Accelerometer"+"/acc_"+act+"_thigh.csv"
        data=pd.read_csv(file_acc,index_col=0)
        title="Subject "+str(sub)+" - "+act+" Accelerometer Data"
        ylabel="Acceleration"
        plot_data(data.attr_time,data.attr_x,data.attr_y,data.attr_z,ylabel,title)
        file_gyro="./Raw_Sensor_Data"+"/Subject_"+str(sub)+"/Gyroscope"+"/Gyroscope_"+act+"_thigh.csv"
        data=pd.read_csv(file_gyro,index_col=0)
        title="Subject "+str(sub)+" - "+act+" Gyroscope Data"
        ylabel="Angular Velocity"
        plot_data(data.attr_time,data.attr_x,data.attr_y,data.attr_z,ylabel,title)