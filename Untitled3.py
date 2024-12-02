#!/usr/bin/env python
# coding: utf-8

# # translation then rotation then reflection then rotation

# In[2]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
img = input("image path")


src_image = cv2.imread(img)


h, w, _ = src_image.shape
T = np.float32([[1, 0, 100], [0, 1, 200]])
translated_img = cv2.warpAffine(src_image, T, (w, h))



cr = (w // 2, h // 2)
R_60 = cv2.getRotationMatrix2D(cr, 60, 1)
rotate_60 = cv2.warpAffine(translated_img, R_60, (w, h))


cr = (w // 2, h // 2)
Rx = np.float32([[1, 0, 0], [0, -1, h]])
X_axis_Rot = cv2.warpAffine(rotate_60, Rx, (w, h))


cr = (w // 2, h // 2)
R_45 = cv2.getRotationMatrix2D(cr, 45, 1)
rotate_45 = cv2.warpAffine(X_axis_Rot, R_45, (w, h))


plt.imshow(cv2.cvtColor(rotate_45, cv2.COLOR_BGR2RGB))
plt.show()


# # translation then rotation then sharing on x axis by 0.2 and then reflection on y-axis

# In[4]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

img = input("Image path: ")

src_image = cv2.imread(img)



h, w, _ = src_image.shape

T = np.float32([[1, 0, 100], [0, 1, 300]])
translated_img = cv2.warpAffine(src_image, T, (w, h))

cr = (w // 2, h // 2)
R_45 = cv2.getRotationMatrix2D(cr, 45, 1)
rotate_45 = cv2.warpAffine(translated_img, R_45, (w, h))

shearing_matrix = np.float32([[1, 0.2, 0], [0, 1, 0]])
sheared_img = cv2.warpAffine(rotate_45, shearing_matrix, (w, h))

reflection_matrix = np.float32([[-1, 0, w], [0, 1, 0]])  # Flip along Y-axis
reflected_img = cv2.warpAffine(sheared_img, reflection_matrix, (w, h))

plt.imshow(cv2.cvtColor(reflected_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()


# # translation then rotation then scaling on x-axis or y-axis then reflection

# In[5]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Input image path
img = input("Image path: ")

# Read the image
src_image = cv2.imread(img)

h, w, _ = src_image.shape

# Step 1: Translation matrix (shifting image by 100 pixels right and 200 pixels down)
T = np.float32([[1, 0, 100], [0, 1, 100]])
translated_img = cv2.warpAffine(src_image, T, (w, h))

# Step 2: Rotate the image 30 degrees around the center
cr = (w // 2, h // 2)
R_30 = cv2.getRotationMatrix2D(cr, 30, 1)
rotated_img = cv2.warpAffine(translated_img, R_30, (w, h))

# Step 3: Scaling transformation (Sx=2, Sy=0.5)
scaling_matrix = np.float32([[2, 0, 0], [0, 0.5, 0]])
scaled_img = cv2.warpAffine(rotated_img, scaling_matrix, (int(w * 2), int(h * 0.5)))

# Step 4: Reflection by flipping on Y-axis
reflection_matrix = np.float32([[-1, 0, scaled_img.shape[1]], [0, 1, 0]])  # Flip along Y-axis
reflected_img = cv2.warpAffine(scaled_img, reflection_matrix, (scaled_img.shape[1], scaled_img.shape[0]))

# Display the final image
plt.imshow(cv2.cvtColor(reflected_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()


# In[ ]:




