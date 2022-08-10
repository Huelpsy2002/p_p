from sklearn import preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Preprrocessing:
    filepath = 'D:\python\\file9.csv'
   
    #------ ------------method to read the csv file --------------------- 
    def read_csv_data(self):
        csvfile = pd.read_csv(self.filepath,header=0)
        df = pd.DataFrame(csvfile)

        print(df)   
                
        
       
       
    # -------------------------Normalization ---------------
    # -------------------------min_max_scaler method----------------
    def min_max_scaler (self) :
        csvfile = pd.read_csv(self.filepath,header=0)
        min_max = preprocessing.MinMaxScaler()
        col = csvfile.columns
        result = min_max.fit_transform(csvfile)
        min_max_scale_df = pd.DataFrame(result,columns=col)
        print(min_max_scale_df)

    # -------------------------- z-score method ----------------------------
    def z_score(self)  :
        csvfile = pd.read_csv(self.filepath,header=0)
        df = pd.DataFrame(csvfile)
        for colums in df.columns:
            df[colums] = (df[colums]-df[colums].mean())/df[colums].std()
        print(df) 
    

    # ---------------------outlier detection --------------------------
    # ---------------------(IQR) method -------------------------------
    def IQR(self):
        csvfile = pd.read_csv(self.filepath)
        df = pd.DataFrame(csvfile)
        saved_column = df['Value']
        plt.plot(saved_column)
        plt.show()                  # <----show plot of [Value] colume  of the csv file before removing outliers
        print('dataset before removing outliers')
        print(df)
        Q1 = df['Value'].quantile(.25)    # |
        Q3 = df['Value'].quantile(.75)    # |<-----Implemntation of IQR method
        q1 = Q1-1.5*(Q3-Q1)               # |
        q3 = Q3+1.5*(Q3-Q1)               # |

        df1 = df[df['Value'].between(q1, q3)]
        saved_column = df1['Value']
        plt.plot(saved_column)
        plt.show()
        print('dataset after removing outliers')
        print(df1)

    #-------------------------z_score_outlier method-------------------------
    def z_score_outlier(self):
        csvfile = pd.read_csv(self.filepath)
        df = pd.DataFrame(csvfile)
        df1 = list(df['Value'])

        saved_column = df['Value']
        plt.plot(saved_column,'o')
        plt.show() 
        mean = np.mean(df1)
        std = np.std(df1)

        threshold = 1
        outlier = []
        for i in df1:
            z = (i-mean)/std
            if z > threshold:
                outlier.append(i)
                df1.remove(i)        # it will remove 500 from [Value] column as outlier
        print(f'outliers = {outlier}') 
        plt.plot(df1)
        plt.show()

        



#-------------------make object f from the class and running the methods of the class--------------------
f = Preprrocessing()
f.read_csv_data()
# f.min_max_scaler()
# f.z_score()
# f.IQR()
# f.z_score_outlier()

