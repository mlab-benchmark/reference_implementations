import numpy as np
import os
import pathlib
import threading
import time
import sys
import argparse
import csv
import matplotlib.pyplot as plt








# only used if script is run as 'main' from command line
def main():

	#construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()  
	ap.add_argument('-f', '--file', type = str, default = 'metrics_resnet50.csv', help = 'csv file with metrics calculated')

	args = ap.parse_args()  
  
	print('Command line options:')
	print(' --file   : ', args.file)


	metrics = []
	numThreads = []
	fps = []
	runtime = []


	with open(args.file, 'r') as csvFile:

		csvReader = csv.reader(csvFile, delimiter=',')
  

		# extracting field names through first row
		fields = next(csvReader)
  
		# extracting each data row one by one
		for row in csvReader:
			numThreads.append(int(row[0]))
			fps.append(int(row[1]))
			runtime.append(float(row[2]))

		
	# printing the field names
	print('Field names are: ' + ', '.join(field for field in fields))


	metrics = [numThreads, fps, runtime]

	#Select the name after 'metrics_csv'
	title = args.file[8: ]

	#Select the name before '.csv'
	title = title.partition('.')
	title = title[0]




	figure = plt.figure()

	# plotting the points
	plt.plot(metrics[0], metrics[2])
 
	# naming the x axis
	plt.xlabel('Number of threads')

	# naming the y axis
	plt.ylabel('Runtime')

 
	# giving a title to my graph
	plt.title(f'UNet with {title} backbone')

	plt.savefig(f'{title}_runtime.png')

	plt.close(figure)




	figure = plt.figure()

	# plotting the points
	plt.plot(metrics[0], metrics[1])
 
	# naming the x axis
	plt.xlabel('Number of threads')

	# naming the y axis
	plt.ylabel('Frames per second')
 
	# giving a title to my graph
	plt.title(f'UNet with {title} backbone')

	plt.savefig(f'{title}_fps.png')

	plt.close(figure)


if __name__ == '__main__':
	main()