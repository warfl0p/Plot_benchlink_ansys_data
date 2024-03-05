import numpy as np


cake_diam=40/100
cake_height=8/100
cake_thickness=2/1000

cake_vol=( np.pi*((cake_diam+2*cake_thickness)**2)/4-np.pi*(cake_diam**2)/4)*cake_height + np.pi*(cake_diam**2)/4*cake_thickness

fins_height=75/1000
fins_vol=0.001297172
fins_area=fins_vol/fins_height
oil_vol=5/1000
oil_height=0.04
x=1

while x==1:
    oil_vol_temp=0.2**2*np.pi*oil_height-fins_area*oil_height
    if abs(oil_vol-oil_vol_temp)<=0.00001:
        x=0
    elif oil_vol-oil_vol_temp<=0:
        oil_height=oil_height-0.0001
    else:
        oil_height=oil_height+0.0001

print(oil_height)
print(oil_vol_temp)
print(fins_area*oil_height)