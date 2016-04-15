import pandas as pd
import csv

'''
Take a vector as an input and image as key and stores in a CSV file.
@param key:		store product key to which the vector belongs.
@param vector: 	To be stored in csv file
@param flag:	a - to append to file
				w - to overwrite
@param csvFile: file in which the vector needs to be added.
'''
def vec2csv(key, vector, flag, csvFile):
	df = pd.DataFrame(data=[[key, vector]], columns=["key","vector"])
		
	if flag =="a":
		dfile = pd.read_csv(csvFile, index_col=0)
		df = dfile.append(df)
		
	df.to_csv(csvFile)

'''
Take a CSV file as an input and get content as {key: value} pair, key is product id and value is feature vector.
@param csvFile: file in which will be read and converted to key and vector dictonary
'''
def csv2vec(csvFile):
	df = pd.read_csv(csvFile, index_col=0)
	map = dict() 
	for row in df.itertuples():
		map[row.key] = row.vector.replace(' ','').replace('[','').replace(']','')
	for key in map:
		print map[key].split(',')

#vec2csv("im1122" , [5,10,10] , "a" , "W:\\Contests\\Myntra Hacakthon\\test3.csv")
csv2vec("W:\\Contests\\Myntra Hacakthon\\test3.csv")