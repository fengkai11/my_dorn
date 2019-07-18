from scipy import misc
f = misc.face()
misc.imsave('face.png',f)
import matplotlib.pyplot as plt
plt.imshow(f)
plt.axis('off')
plt.show()