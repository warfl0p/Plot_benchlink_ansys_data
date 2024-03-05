import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import re

filetype='pdf'                                          # 'pdf' is betere qualiteit
inputFolder='Sunflower'

## bekijk waar data begint en maak vanaf die rij een dataframe aan
def readData(csvfile):
    result_list=[]
    df = pd.read_csv(csvfile, nrows=3)
    input_string = df.iloc[1, 0]
    for text in re.findall('"([^"]*)"', input_string):
        result_list.append(text)
    df = pd.read_csv(csvfile,skiprows=3,names=result_list, delim_whitespace=True)
    return df

## gaat over alle csvfiles in de datalogger_files map en maakt een dictionairy aan met alle dataframes van die csvfiles er in 
def csvToDf():
    folder_path = f'Resultaten HPC\{inputFolder}'   # in deze folder zitten alle .out files in
    try:
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.out')] 
    except:
        print('ERROR: gekozen comparison folder bestaat niet of staat niet op juiste plaats (variabele: inputFolder)')
    dfs = [readData(os.path.join(folder_path, file)) for file in csv_files]
    df = pd.concat([dfs[0]] + [df.iloc[:, 1] for df in dfs[1:]], axis=1)
    cols = df.columns.tolist()                          #locatie van 2de en 3de column veranderen
    cols[1], cols[2] = cols[2], cols[1]
    df = df[cols]
    return df

## plot van alle dataframes (alle csvfile in datalogger_files dus) de gekozen thermocouples 
def plotMeasurement(fileName):
    df=csvToDf()
    fig, ax = plt.subplots(1)
    ax.grid(True)
    fig.set_size_inches(18.5, 10.5, forward=True)
    try:
        for i in range(2,18):
            plt.plot(df['flow-time'], df.iloc[:,[i]],label=df.columns[i])

            # plt.plot(df['flow-time'], df.iloc[:,[3]],label=df.columns[3])  #thermocouples kiezen
            # plt.plot(df['flow-time'], df.iloc[:,[4]],label=df.columns[4])
            # plt.plot(df['flow-time'], df.iloc[:,[5]],label=df.columns[5])
    except:
        print('ERROR: gekozen columns zitten niet in dataFrame')
    plt.xlabel("Time (s)")                              # as titels toevoegen
    plt.ylabel("Temperature (K)")
    plt.legend()
    # ax.set_ylim(ymin=15)                              # startwaarde op y-as instellen (optioneel)
    fileName=fileName.split('%')                        # neemt text tussen twee '%', anders volledige naam v file als titel en fileName van grafiek
    if len(fileName) >= 3:
        modifiedFileName=fileName[1]
    else:
        modifiedFileName=fileName[0][:-4]
    plt.title(inputFolder)
    plt.savefig(f'plots-sim-{filetype}/{inputFolder}/{modifiedFileName}.{filetype}')
    
## maakt folders aan in plots-sim-pdf of plots-sim-png met dezelfde naam als in Resultaten HPC, alles wordt dan opgeslaan in een folder met dezelfde naam als gekozenfolder      
def createSaveFolders():
    folders = os.listdir('Resultaten HPC')               # Get a list of all the folders in the source directory
    for folder in folders:                               # Loop through the folders and create a new folder with the same name in the target directory
        folder_path = os.path.join(f'plots-sim-{filetype}', folder)
        os.makedirs(folder_path, exist_ok=True)
        
createSaveFolders()


plotMeasurement(inputFolder)
