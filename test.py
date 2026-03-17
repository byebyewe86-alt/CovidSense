import pandas as pd

data = pd.read_csv("covid_data.csv")
#print(data) #print the entire dataset
#print(data.head()) #print the first 5 rows of the dataset
#print(data.info()) #print the information about the dataset

print(data.columns)


#print(data["3/1/20"].plot(kind="hist")) #plot a histogram of the data for the date 3/1/20