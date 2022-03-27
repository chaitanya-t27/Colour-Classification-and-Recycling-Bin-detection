import os, cv2
import numpy as np
from roipoly import RoiPoly
from matplotlib import pyplot as plt
import time
start = time.time()

folder = 'data/validation'
filename = '0061.jpg'

img = cv2.imread(os.path.join(folder,filename))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

mu = np.array([[0.12305833, 0.26567634, 0.64023492],
[0.42882381, 0.57094613, 0.75512574],
[0.28762304, 0.34534259, 0.21421107],
[0.54224106, 0.43418504, 0.38058496]])

var = np.array([[0.01222447, 0.02025565, 0.03034596],
[0.06225271, 0.05990138, 0.05171289],
[0.03851334, 0.03714906, 0.01906854],
[0.04964767, 0.04069557, 0.03690298]])

theta = np.array([0.2616719792141001, 0.2091179385530228, 0.23882356092465112, 0.29038652130822595]).reshape(-1,1)

def pdfunc(x, mu, var):
    return np.exp((-1*(x-mu)**2)/(2*var))*(1/(2*3.14*var)**0.5)


maxap = np.zeros((4,1))
X_flat = img.reshape(img.size//3,3)/255
y = np.zeros((X_flat.shape[0],1))
for k in range(y.shape[0]):   
    for i in range(4): # i is for class number
        prob = np.prod(pdfunc(X_flat[k], mu[i], var[i]))
        maxap[i] = prob * theta[i]
    out = np.argmax(maxap) + 1
    y[k] = out

bin = y.reshape((img.shape[0], img.shape[1],1))
bin[bin > 1] = 0
end = time.time()
print(end-start)
plt.imshow(bin)
plt.show()
