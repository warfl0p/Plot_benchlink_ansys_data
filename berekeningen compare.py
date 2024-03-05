import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

oil_vol=0.005
oil_density=747.807 #[kg/m^3]
# oil_density=920.807 #[kg/m^3]
oil_mass = oil_density*oil_vol # mass in kg

alu_density=2700
alu_c=0.921

fins_vol=0.001297172 
fins_mass=alu_density*fins_vol
cake_diam=40/100
cake_height=8/100
cake_thickness=2/1000
cake_vol=( np.pi*((cake_diam+2*cake_thickness)**2)/4-np.pi*(cake_diam**2)/4)*cake_height + np.pi*(cake_diam**2)/4*cake_thickness
cake_mass=alu_density*cake_vol
c_steel=0.461
steel_density=7700 

water_c= 4.184 
# coefficients of specific heat capacity equation
a = 0.76e-9
b = -3.31e-7
c = 4.147e-05
d = 1.2e-03
e = 1.9506
print(os.getcwd())
# df_no = pd.read_csv('figuren/no fins average.csv')
df_fins = pd.read_csv('figuren/with fins df.csv')
df_no = pd.read_csv('figuren/no fins df.csv')

# df = pd.read_excel('figuren/meting 3 dataframe.xls')

def c_integral(T1,T2):
    T1,T2 = T1 + 273.15, T2 + 273.15
    return (a/5*T2**5 + b/4*T2**4 + c/3*T2**3 + d/2*T2**2 + e*T2) - (a/5*T1**5 + b/4*T1**4 + c/3*T1**3 + d/2*T1**2 + e*T1)

# def c_column():
#     T=df['rolling_average']+273.15
#     df['c'] = a/5*T**5 + b/4*T**4 + c/3*T**3 + d/2*T**2 + e*T
#     df['c_derivative'] = np.gradient(df['c'], 10)
    
def c_column(df):
    T=df['rolling_average']+273.15
    df['c'] = a*T**4 + b*T**3 + c*T**2 + d*T + e
    df['c_derivative'] = np.gradient(df['c'], 10)
    
def calc_oil_q(start_temp,end_temp):
    oil_q = oil_mass * c_integral(start_temp,end_temp)
    # print(f"oil= {oil_q:.2f} kJ")
    return oil_q

def calc_total_q(start_temp,end_temp,boolfins):
    diff_temp= end_temp-start_temp
    oil_q = oil_mass * c_integral(start_temp,end_temp)
    # oil_q = oil_mass * 2.2550976 * diff_temp
    fins_q = 0
    if boolfins==1:
        fins_q = fins_mass * alu_c * diff_temp
    dish_q = cake_mass * alu_c * diff_temp
    total_q=sum([fins_q,dish_q,oil_q])
    # print('total energy=',total_q,'kJ')
    return total_q

def calc_total_q_dot(c, dT,boolfins):
    oil_q = oil_mass * c * dT
    fins_q = 0
    if boolfins==1:
        fins_q = fins_mass * alu_c * dT
    dish_q = cake_mass * alu_c * dT
    total_q=sum([fins_q,dish_q,oil_q])
    # print('total energy=',total_q,'kJ')
    return total_q

def calc_water_mass(start_oil,end_oil,start_water,end_water,boolfins):
    total_q=calc_total_q(start_oil,end_oil,boolfins)
    diff_temp=end_water-start_water
    water_mass=-total_q/water_c/diff_temp
    # print('kokend water=',water_mass,'kg')
    return water_mass

def plot_water_mass():
    #  print(os.getcwd())
    df_fins['waterkoken'] = df_fins['average'].apply(lambda x: calc_water_mass(x,100,20,100,1))
    df_no['waterkoken'] = df_no['average'].apply(lambda x: calc_water_mass(x,100,20,100,0))
    fig, ax1 = plt.subplots()
    # df.to_excel('df in excel.xlsx')
    ax1.plot(df_no["Time_seconds"],df_no['waterkoken'], label='no fins')
    ax1.plot(df_fins["Time_seconds"],df_fins['waterkoken'], label='with fins')
    ax1.set_xlabel("Time (s)")  
    ax1.set_ylabel('Volume (l)')
    ax1.legend()
    plt.savefig('figuren/plot_water_mass.pdf',bbox_inches='tight')
    plt.show()

def plot_total_q():
    df_no['total q'] = df_no['average'].apply(lambda x: calc_total_q(20,x,0))
    df_fins['total q'] = df_fins['average'].apply(lambda x: calc_total_q(20,x,1))
    fig, ax1 = plt.subplots()
    # df.to_excel('df in excel.xlsx')
    ax1.plot(df_no["Time_seconds"],df_no['total q'], label='no fins')
    ax1.plot(df_fins["Time_seconds"],df_fins['total q'], label='with fins')
    ax1.set_xlabel("Time (s)")  
    ax1.set_ylabel('Total heat (J)')
    ax1.legend()
    # plt.savefig('plot_total_q.pdf',bbox_inches='tight')
    plt.savefig('figuren/plot_total_q.pdf',bbox_inches='tight')
    plt.show()

def plot_total_q_dot():
    c_column(df_fins)
    c_column(df_no)
    # df['total qdot'] = df['Time_derivative'].apply(lambda x: calc_total_q(20,x))
    df_fins['total qdot'] = calc_total_q_dot(df_fins['c'],df_fins['Time_derivative'],1 )
    x_time_fins=df_fins["Time_seconds"][400:]
    df_no['total qdot'] = calc_total_q_dot(df_no['c'],df_no['Time_derivative'],0 )
    x_time_no=df_no["Time_seconds"][400:]
    fig, ax1 = plt.subplots()
    # df.to_excel('df in excel.xlsx')
    ax1.plot(x_time_no,df_no['total qdot'][400:], label='no fins')
    ax1.plot(x_time_fins,df_fins['total qdot'][400:], label='with fins')
    ax1.set_xlabel("Time (s)")  
    ax1.set_ylabel('Total heat exchange (kJ/s)')
    ax1.legend()
    # plt.savefig('plot_total_q_dot.pdf',bbox_inches='tight')
    plt.savefig('figuren/plot_total_q_dot.pdf',bbox_inches='tight')
    plt.show()
    
# plot_water_mass()
# plot_total_q()
plot_total_q_dot()
    




## constant heat cap--
def specific_heat_capacity(T):
        return a*T**4 + b*T**3 + c*T**2 + d*T + e
c_const=specific_heat_capacity((20+200)/2)
print(c_const)
Q_const = oil_mass * c_const * (200-20)
##
