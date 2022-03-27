'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''


import os, cv2
import numpy as np
from roipoly import RoiPoly
from matplotlib import pyplot as plt


if __name__ == '__main__':

  # read the first training image
  folder = 'data/training'
  Xg = np.empty((0,3))
  for filename in os.listdir(folder):
    # filename = '0001.jpg'  
    img = cv2.imread(os.path.join(folder,filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  
    # display the image and use roipoly for labeling
    fig, ax = plt.subplots()
    ax.imshow(img)
    my_roi = RoiPoly(fig=fig, ax=ax, color='r')
    
    # get the image mask
    mask = my_roi.get_mask(img)
    mx,my = np.where(mask==1)
    
    # display the labeled region and the image mask
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('%d pixels selected\n' % img[mask,:].shape[0])
    
    ax1.imshow(img)
    ax1.add_line(plt.Line2D(my_roi.x + [my_roi.x[0]], my_roi.y + [my_roi.y[0]], color=my_roi.color))
    ax2.imshow(mask)
    
    plt.show(block=True)

    n = np.size(mx)
    mask_rgb = np.zeros([n,3])
    mask_rgb = img[mx,my].astype(np.float64)/255
    if mask_rgb.shape[0]>200:
      index = np.random.randint(mask_rgb.shape[0], size=200)
      mask_rgb = mask_rgb[index,:]
    # np.save(filename[:-3]+"_binblue", mask_rgb)
    Xg = np.vstack([Xg, mask_rgb])
  np.save('Xgreen',Xg)