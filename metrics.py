import numpy as np

UV = np.loadtxt('melspec_ALS/UV.csv', delimiter=',')
_UV = np.loadtxt('melspec_ALS/UVd.csv', delimiter=',')
positives = 0
for x in np.nditer(UV, op_flags=['readwrite']):
    if x > 0.5:
        x[...] = 1
        positives += 1
    else:
        x[...] = 0
for x in np.nditer(_UV, op_flags=['readwrite']):
    if x > 0.5:
        x[...] = 1
    else:
        x[...] = 0
        
samples = UV.shape[0]*UV.shape[1]
acc = np.sum(UV == _UV)
acc = acc/samples
print("Accuracy is defined as correctly observed preferences of users on both positives and negatives")
print("Accuracy : ",acc)

true_pos = 0
for x,y in zip(UV.flatten(),_UV.flatten()):
    if x == 1 and y ==1:
        true_pos += 1
true_pos /= positives
print("Precision is defined as model correctly recommending songs based on user preference")
print("Precision : ",true_pos)
