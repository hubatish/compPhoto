import numpy as np
arr = np.zeros((720,924,3))
seam = np.zeros(720,dtype='int')# + 5
for i in range(20,50):
    seam[i] = i
#print 'seam[6]',seam[6]
print 'start arr', arr.shape, 'seam shape ',seam.shape

#dArr = np.delete(arr,seam,0)
m,n,derp = arr.shape
#dArr = arr[np.arange(n)!=np.array(seam)[:,None]].reshape(m,-1)

mask = np.ones((m,n),dtype=bool)
mask[range(m),seam] = False
dArr = arr[mask].reshape(m,n-1,3)

#dArr = np.delete(arr[2],8,0)
print 'new arr',dArr.shape

print 'del 20',np.delete(arr,20,0).shape
print 'del 0',np.delete(arr,0,0).shape

