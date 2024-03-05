import pandas as pd
import matplotlib.pyplot as plt
import os

#Deze code maakt grafieken van de gekozen columns=thermocouples (zie bij functie plotMeasurement).
#Alle csv bestanden in de folder datalogger_files map worden gebruikt, dus als hier 10 datalog bestanden in staan worden er 10 grafieken gemaakt.
#Afhankelijk of png of pdf gekozen wordt als datatype wordt alles opgeslaan in plots-pdf of plots-png.

#Als titel (van de plot) en de naam (van het document waardat de plot wordt in opgeslaan) wordt de naam van het csv bestand gebruikt, 
#tenzij er op de volgende manier een titel wordt toegevoegd: 'Data 8199 2391 3_6_2023 -hier kies ik titel van file en graphiek-.csv',
#tussen 2 koppeltekens (-) dus.

##filetype kiezen: 'png' of 'pdf'
filetype='png'      # 'pdf' is betere qualiteit

## bekijk waar data begint en maak vanaf die rij een dataframe aan
def readData(csvfile):
    df = pd.read_csv(csvfile,usecols = [0],encoding='UTF-16 LE')
    rowsToSkip=df.loc[df[df.columns[0]] == 'Scan'].index[0]+1
    df = pd.read_csv(csvfile,skiprows=rowsToSkip,encoding='UTF-16 LE')
    return df

## functie die de alarm colommen en foutieve thermocouples verwijderd uit de df
def dropAlarmCols(df):
    colNames=list(df.columns)                               
    indexRemoveCol=[]
    for i in range(3,len(colNames)):
        if i%2==True:
                indexRemoveCol.append(colNames[i])
    df=df.drop(columns=indexRemoveCol)
    try:
        df=df.drop(columns=['102 (C)','104 (C)','117 (C)','120 (C)'])  #deze thermocouples werken niet en worden dus uit dataframe verwijderd
    except:
        print('ERROR: kan gekozen columns niet droppen omdat ze niet bestaan in dataframe') 
    return df

## functie die de tijd kolom omzet naar looptijd in seconden en verwijderd
def formatTimecol(df):
    df.insert(2, 'Time_seconds', None)                  # voegt colomn looptijd in seconden toe
    df['Time'] = pd.to_datetime(df['Time'], format='%d/%m/%Y %H:%M:%S:%f')
    df["Time_seconds"] = (df['Time'] -df['Time'].iat[0]).apply(lambda x: x.total_seconds() )
    df=df.drop(columns='Time')
    return df

## functie die column toevoegd die de gemiddeldes van alles thermocouples bevat per tijdstip
def avgColumn(df):
    df.insert(2, 'average', None)                       # voegt colomn 'average' toe
    col = df.loc[: , "103 (C)":"106 (C)"]
    df['average'] = col.mean(axis=1)
    return df

## gaat over alle csvfiles in de datalogger_files map en maakt een dictionairy aan met alle dataframes van die csvfiles er in 
def dataframeToDict():
    dfs = {}
    folder_path = 'datalogger_files'                    # hier zitten alle csv files in
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]  
    for file in csv_files:
        file_path = os.path.join(folder_path, file) 
        df = readData(file_path)                        #dataframe uitlezen
        df = dropAlarmCols(df)                          #enkele bewerkingen op dataframe
        df = formatTimecol(df)
        df = avgColumn(df)
        dfs[file] = df                                  # Use the file name as the key and store the DataFrame in the dictionary
    return dfs                                          # returns dictionary met dataframes

## plot van alle dataframes (alle csvfile in datalogger_files dus) de gekozen thermocouples 
def plotMeasurement(fileName,df):
    fig, ax = plt.subplots(1)
    ax.grid(True)
    try:
        plt.plot(df['Time_seconds'], df.iloc[:, [2]],label=df.columns[2])  #thermocouples kiezen
        plt.plot(df['Time_seconds'], df.iloc[:, [2]],label=df.columns[2])  
        plt.plot(df['Time_seconds'], df.iloc[:, [4]],label=df.columns[4])  #ipv df.iloc[:, [6]] kan ook naam van kolom nemen zoals df['104 (C)'] of 'average'
        plt.plot(df['Time_seconds'], df.iloc[:, [6]],label=df.columns[6])  #bij label=... kan '104 (C)' geschreven worden of eender andere naam voor die kolom
    except:
        print('ERROR: gekozen columns zitten niet in dataFrame')
    plt.xlabel("Time [s]")                              # as titels toevoegen
    plt.ylabel("Temperature [K]")
    plt.legend()
    # ax.set_ylim(ymin=15)                              # startwaarde op y-as instellen (optioneel)
    fileName=fileName.split('-')                        # text tussen twee '-' nemen, anders volledige naam v file als titel en fileName van grafiek
    if len(fileName) >= 3:
        plt.title(fileName[1])
        plt.savefig(f'plots-{filetype}/{fileName[1]}.{filetype}')
    else:
        plt.title(fileName[0][:-4])
        plt.savefig(f'plots-{filetype}/{fileName[0][:-4]}.{filetype}') 
                

dfs=dataframeToDict()                                     

for file, df in dfs.items():
    plotMeasurement(file, df)