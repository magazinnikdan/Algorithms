import numpy as np
ss=r'C:\Users\magaz\Documents\22_docadsfasdf.jpg'

ii=ss.rfind("\\")
ss2=ss[ii+1:-4]
ii=ss2.find("_")
ss3=ss2[0:ii]
print(ss3)
#
# kk=np.zeros(10)
# print(kk)