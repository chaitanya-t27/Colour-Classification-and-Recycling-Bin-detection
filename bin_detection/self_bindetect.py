import os, cv2
import numpy as np
from roipoly import RoiPoly
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops
import time
start = time.time()

folder = 'data/validation'
filename = '0067.jpg'

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
        prob = 1
        for j in range(3): # j is for r/g/b
            prob *= pdfunc(X_flat[k][j], mu[i][j], var[i][j])
        maxap[i] = prob * theta[i]
    out = np.argmax(maxap) + 1
    y[k] = out

bin = y.reshape((img.shape[0], img.shape[1]))
bin[bin > 1] = 0
mask_img = bin 
end = time.time()
print(end-start)
plt.imshow(mask_img)
plt.show()

mask_labels = label(np.asarray(bin))
props = regionprops(mask_labels)
boxes = []
img_copy = np.asarray(bin)
for prop in props:
    if (prop.bbox_area > 1000):
        cv2.rectangle(img_copy, (prop.bbox[1], prop.bbox[0]),(prop.bbox[3], prop.bbox[2]),(150,200,100), 2)
fig, (ax2, ax3) = plt.subplots(1, 2, figsize = (10, 5))
ax3.set_title('Image with derived bounding box')
ax2.imshow(bin)#, cmap='gray')
#plt.show()
boxes.append([prop.bbox[1],prop.bbox[0],prop.bbox[3],prop.bbox[2]])
print(boxes)
cv2.imshow('image',bin)
cv2.waitKey(10000)