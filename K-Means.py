#import libraries
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import time
import matplotlib.pyplot as plt

# Import csv file
csv_file = pd.read_csv(r'E:\University\Semester 7\Analysis and Design of Algorithms (CSEN 707)\group3.csv', delimiter=',', header=None, skiprows=1, names=["Height","Weight","BMI","L_Sh","L_Arm"])
#data.head()
#print(csv_file)

# Get data mean
csv_file_mean = csv_file[["Height","Weight","BMI","L_Sh","L_Arm"]].mean()
#print(csv_file_mean)

# Get data standard deviation
csv_file_std = csv_file[["Height","Weight","BMI","L_Sh","L_Arm"]].std()
#print(csv_file_std)

# Get data variance
csv_var = csv_file[["Height","Weight","BMI","L_Sh","L_Arm"]].var()
#print(csv_file_var)

# Normalizing the data
data = ((csv_file-csv_file_mean)/csv_file_std).abs()
#print(data)

# K-Mean method (N => samples & K => groups) 
def K_Mean(N,K):
    # Run time
    global running_time
    difference = 1
    j = 0
    # Centers (Average of group of data)
    Centers = (N.sample(n=K))

    while(difference != 0):
        # Start time
        starting_time = time.time()
        N_1 = N
        i = 1
        # For every entry sampled data point
        for index_1,row_c in Centers.iterrows():
            tmp = []
            # For every data point get distance between this center and every point
            for index_2,row_d in N_1.iterrows():
                d_height = (row_c["Height"] - row_d["Height"]) ** 2
                d_weight = (row_c["Weight"] - row_d["Weight"]) ** 2
                d_bmi = (row_c["BMI"] - row_d["BMI"]) ** 2
                d_L_Shoulder = (row_c["L_Sh"] - row_d["L_Sh"]) ** 2
                d_L_Arm = (row_c["L_Arm"] - row_d["L_Arm"]) ** 2
                d_all = np.sqrt(d_height+d_weight+d_bmi+d_L_Shoulder+d_L_Arm)
                tmp.append(d_all)
            # Store distance between center and every point in a column
            N[i] = tmp
            i += 1

        tmp_1 = []
        # To get min distance and its position
        ## For each row in the data set get
        ### To get min of each row between Center 1,2,3 
        
        # for index, row in N.iterrows():
        #         min_distance = row[1]
        #         position = 1
        #         for i in range(K):
        #             # Check if the next row is less than the current min row
        #             if row[i + 1] < min_distance:
        #                 min_distance = row[i + 1]
        #                 position = i + 1
        #             # Save position of min distance
        #         tmp_1.append(position)
        
        # To get min of each row between Center 1,2,3
        tmp_1 = N[[1,2,3]].idxmin(axis=1)
        
        # Store the mins in group called cluster
        N["Cluster"] = tmp_1
        
        # Regroup centers into cluster as rows. Get the mean of every column
        Centers_new = N.groupby(["Cluster"]).mean()[["Height","Weight","BMI","L_Sh","L_Arm"]]
        
        # Get the loop running until difference reaches its minimum or near 0
        if (j == 0):
                difference = 1
                j += 1
        else:
                # Subtract new and old centers to get the main difference
                difference_height = (Centers_new['Height'] - Centers['Height']).sum()
                difference_weight = (Centers_new['Weight'] - Centers['Weight']).sum()
                difference_bmi = (Centers_new['BMI'] - Centers['BMI']).sum()
                difference_L_Sh = (Centers_new['L_Sh'] - Centers['L_Sh']).sum()
                difference_L_Arm = (Centers_new['L_Arm'] - Centers['L_Arm']).sum()
                difference = difference_height + difference_weight + difference_bmi + difference_L_Sh + difference_L_Arm
                difference = abs(difference)
        Centers = N.groupby(["Cluster"]).mean()[["Height","Weight","BMI","L_Sh","L_Arm"]]
    running_time.append(time.time() - starting_time)

# Choose number of N & K that are required to be applied
N = [100,200,500,1000,2000,5000]
K = [3,5]
for k in K:
    running_time = []
    for n in N:
        tmp_data = data.iloc[:n]
        N1 = tmp_data[["Height", "Weight", "BMI", "L_Sh", "L_Arm"]]
        K_Mean(N1,k)
    
    # Plotting
    plt.plot(N,running_time)
    plt.xlabel('N Different Values')
    plt.ylabel('Run Time')
    plt.title(f'Run Time @ K = {k}')
    plt.show()