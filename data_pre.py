import csv
import numpy as np 
from scipy.misc import imread

lines = []

with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
lines = lines[1:]

images = []
measurements = []
for line in lines:
	filename=line[0].split('/')[-1]
	current_path = 'data/IMG/' + filename
	# image = cv2.imread(current_path)/255.0-0.5
	image = imread(current_path)
	images.append(image)
	measurements.append(float(line[3]))

print("Data preparation finished.")

X_train = np.array(images)
y_train = np.array(measurements)

# np.round(X_train,decimals=3)
# np.round(y_train,decimals=3)

np.save('data/X_train.npy', X_train)
print("Training data is saved.")


np.save('data/y_train.npy', y_train)
