import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

oil_vol=0.005
oil_density=920.807 #[kg/m^3]
# oil_density=747.807
oil_mass = oil_density*oil_vol # mass in kg
print(5.150-oil_mass)
print( 8.362-5.150)
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
# df_average = pd.read_csv('figuren/no fins average.csv')
df_average = pd.read_csv('figuren/no fins df.csv')
# df_average = pd.read_csv('no fins df.csv')
df=df_average
# df = pd.read_excel('figuren/meting 3 dataframe.xls')

def c_integral(T1,T2):
    T1,T2 = T1 + 273.15, T2 + 273.15
    return (a/5*T2**5 + b/4*T2**4 + c/3*T2**3 + d/2*T2**2 + e*T2) - (a/5*T1**5 + b/4*T1**4 + c/3*T1**3 + d/2*T1**2 + e*T1)

# def c_column():
#     T=df['rolling_average']+273.15
#     df['c'] = a/5*T**5 + b/4*T**4 + c/3*T**3 + d/2*T**2 + e*T
#     df['c_derivative'] = np.gradient(df['c'], 10)
    
def c_column():
    T=df['rolling_average']+273.15
    df['c'] = a*T**4 + b*T**3 + c*T**2 + d*T + e
    df['c_derivative'] = np.gradient(df['c'], 10)
    
def calc_oil_q(start_temp,end_temp):
    oil_q = oil_mass * c_integral(start_temp,end_temp)
    # print(f"oil= {oil_q:.2f} kJ")
    return oil_q

def calc_total_q(start_temp,end_temp):
    diff_temp= end_temp-start_temp
    oil_q = oil_mass * c_integral(start_temp,end_temp)
    # oil_q = oil_mass * 2.2550976 * diff_temp
    fins_q = fins_mass * alu_c * diff_temp
    dish_q = cake_mass * alu_c * diff_temp
    total_q=sum([fins_q,dish_q,oil_q])
    # print('total energy=',total_q,'kJ')
    return total_q

def calc_total_q_dot(c, dT):
    oil_q = oil_mass * c * dT
    fins_q = 0
    fins_q = fins_mass * alu_c * dT
    dish_q = cake_mass * alu_c * dT
    total_q=sum([fins_q,dish_q,oil_q])
    # print('total energy=',total_q,'kJ')
    return total_q

def calc_water_mass(start_oil,end_oil,start_water,end_water):
    total_q=calc_total_q(start_oil,end_oil)
    diff_temp=end_water-start_water
    water_mass=-total_q/water_c/diff_temp
    # print('kokend water=',water_mass,'kg')
    return water_mass

def plot_water_mass():
    #  print(os.getcwd())
    df['waterkoken'] = df['average'].apply(lambda x: calc_water_mass(x,100,20,100))
    fig, ax1 = plt.subplots()
    # df.to_excel('df in excel.xlsx')
    # ax1.plot(df["Time_seconds"],df['waterkoken'])
    seconds_weg=2740
    df2=df[(df['waterkoken'] > 0) & (df["Time_seconds"] > seconds_weg)]
    ax1.plot(df2["Time_seconds"]-seconds_weg,df2['waterkoken'].dropna())
    ax2 = ax1.twinx()
    ax2.plot(df2["Time_seconds"]-seconds_weg,df2['average']+273.15,color='#ff7f0e')
    ax1.set_xlabel("Time (s)")  
    ax1.set_ylabel('Volume (l)')
    ax2.set_ylabel('Temperature (K)')
    handles, labels = [], []
    handles.append(ax1.lines[0])
    handles.append(ax2.lines[0])
    labels.append('Amount of water')
    labels.append('Temperature oil')
    ax1.legend(handles, labels)
    # plt.savefig('plot_water_mass.pdf',bbox_inches='tight')
    plt.savefig('figuren/plot_water_mass.pdf',bbox_inches='tight')
    plt.show()
    return None

def plot_total_q():
    #  print(os.getcwd())
    print(df)
    df['total q'] = df['average'].apply(lambda x: calc_total_q(20,x))
    fig, ax1 = plt.subplots()
    # df.to_excel('df in excel.xlsx')
    ax1.plot(df["Time_seconds"],df['total q'])
    ax2 = ax1.twinx()
    ax2.plot(df["Time_seconds"],df['average']+273.15,color="#ff7f0e")
    ax1.set_xlabel("Time (s)")  
    ax1.set_ylabel('Total heat (J)')
    ax2.set_ylabel('Temperature (K)')
    handles, labels = [], []
    handles.append(ax1.lines[0])
    handles.append(ax2.lines[0])
    labels.append('Total Heat')
    labels.append('Temperature oil')
    ax1.legend(handles, labels)
    # plt.savefig('plot_total_q.pdf',bbox_inches='tight')
    plt.savefig('figuren/plot_total_q.pdf',bbox_inches='tight')
    plt.show()
    return None

def plot_total_q_dot():
    c_column()
    begin_row=50
    end_row=150
    # df['total qdot'] = df['Time_derivative'].apply(lambda x: calc_total_q(20,x))
    dt = 10  # seconds
    df['rolling_average'] = df['average'].rolling(window=20).mean()
    df['temperature_derivative'] = df['rolling_average'].diff() / dt
    df['total qdot'] = calc_total_q_dot(df['c'],df['temperature_derivative'] )
    x_time=df["Time_seconds"][begin_row:end_row]
    fig, ax1 = plt.subplots()
    # df.to_excel('df in excel.xlsx')
    ax1.plot(x_time,df['total qdot'][begin_row:end_row])
    ax2 = ax1.twinx()
    ax2.plot(x_time,df['average'][begin_row:end_row]+273.15,color="#ff7f0e")
    ax1.set_xlabel("Time (s)")  
    ax1.set_ylabel('Heat transfer rate (kJ/s)')
    ax2.set_ylabel('Temperature (K)')
    handles, labels = [], []
    handles.append(ax1.lines[0])
    handles.append(ax2.lines[0])
    labels.append('Cooling rate')
    labels.append('Temperature oil')
    ax1.legend(handles, labels, loc='center right')
    # plt.savefig('plot_total_q_dot.pdf',bbox_inches='tight')
    plt.savefig('figuren/plot_total_q_dot.pdf',bbox_inches='tight')
    df.to_csv('temp.csv',index=False)
    plt.show()
    x = x_time.values.reshape(-1, 1)  # reshape(-1, 1) converts a 1D array to a 2D array
    y = df['total qdot'][begin_row:end_row].values
    model = LinearRegression()
    model.fit(x, y)
    a = model.coef_[0]*1000
    b = model.intercept_*1000
    print(f'y = {a}x + {b}')
    plt.scatter(x, y)
    plt.plot(x, model.predict(x), color='red')
    plt.show()
    return None
    
plot_water_mass()
plot_total_q()
plot_total_q_dot()
    




## constant heat cap--
def specific_heat_capacity(T):
        return a*T**4 + b*T**3 + c*T**2 + d*T + e
c_const=specific_heat_capacity((20+200)/2)
print(c_const)
Q_const = oil_mass * c_const * (200-20)
##
