## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
  • Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  ```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
  ```
![image](https://github.com/user-attachments/assets/2a755835-6a07-48ef-9913-be8a05224283)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/a63314c8-921a-4ff2-bf11-8cb4be6caf4a)
```
df["Highly Positive Skew"]=np.log(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/c50b8b84-78d5-4661-b137-02f8936a714a)
```
df["Moderate Positive Skew"]=np.reciprocal(df["Moderate Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/c82be827-8bf2-4a7e-82f4-360a160c6547)
```
df["Highly Negative Skew"]=np.sqrt(df["Highly Negative Skew"])
df
```
![image](https://github.com/user-attachments/assets/71a33bdf-4496-45c8-9771-2213011ccfad)
```
df["Highly Positive Skew"]=np.square(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/c166bfb3-5124-4a1b-978a-77399b5ebe32)
```
df.skew( )
```
![image](https://github.com/user-attachments/assets/6c1c0a6c-ddb9-4fad-9e7f-859e6bdb3fb8)
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method
```
df["Highly Positive Skew_boxcox"],parameter=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/2eb86b83-6565-419e-b4eb-311aa6a3aac5)
```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df
```
![image](https://github.com/user-attachments/assets/adc3946b-6945-4b5d-a24e-18cf7dbdcde3)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
```
```
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/f4018b44-97ec-4bed-a27c-96e892ef8456)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/51301584-76e6-4814-8590-252f75c2e0e8)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
```
```
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
```
```
sm.qqplot(df["Moderate Negative Skew"],line='45')
```
![image](https://github.com/user-attachments/assets/074767ee-9fff-4600-9a6e-967bcd7a3810)
![image](https://github.com/user-attachments/assets/7767b1ec-e7a1-4f25-b7a6-da1812a09248)


# RESULT:
       Thus perform Feature Encoding and Transformation process has been done for the given data. 

       
