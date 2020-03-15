import pandas as pd
thigh_data=pd.DataFrame()
window=100
last=0
activities=["climbingdown","climbingup","jumping","lying","running","sitting","standing","walking"]
for sub in range(1,7):
    for i,act in enumerate(activities):
        file_acc="./Raw_Sensor_Data"+"/Subject_"+str(sub)+"/Accelerometer"+"/acc_"+act+"_thigh.csv"
        file_gyro="./Raw_Sensor_Data"+"/Subject_"+str(sub)+"/Gyroscope"+"/Gyroscope_"+act+"_thigh.csv"
        acc=pd.read_csv(file_acc,index_col=0)
        gyro=pd.read_csv(file_gyro,index_col=0)
        if (len(gyro)>len(acc)):
            extra=len(gyro)-len(acc)
            acc=acc[:-extra]
        if (len(acc)>len(gyro)):
            extra=len(acc)-len(gyro)
            gyro=gyro[:-extra]
        data=pd.merge(gyro,acc,right_index=True,left_index=True,suffixes=('_gyro','_acc'))
        data['Activity']=act
        data['Label']=i
        data['Subject']=sub
        data['Sample_Num']=0
        remainder=len(data)%window
        data=data[:-remainder]
        number_samples=int(len(data)/window)
        row_number=0
        for j in range(0,number_samples):
            data.Sample_Num[row_number:row_number+window]=last+j
            row_number=row_number+window
        last+=number_samples
        thigh_data=thigh_data.append(data,ignore_index=True)
training=thigh_data[thigh_data.Subject.between(1,4)]
testing=thigh_data[thigh_data.Subject.between(5,6)]
training.to_csv('./Data/thigh_training.csv')
testing.to_csv('./Data/thigh_testing.csv')