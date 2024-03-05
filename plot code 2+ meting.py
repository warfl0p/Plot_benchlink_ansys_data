import pandas as pd
import matplotlib.pyplot as plt
import os

# Het verschil tussen deze code en 'plot code 1 meting' is dat deze een folder uit de folder datalogger_files haalt
# dit bestand gebruikt enkel de bestanden binnen die folder en vergelijkt de metingen met elkaar, dus als er 3 bestanden in staan wordt
# er 1 grafiek gemaakt die de 3 metingen met elkaar vergelijkt.

# Ook hier moeten thermocouples gekozen worden, voor alle bestanden worden dan dezelfde
# thermocouples vergeleken (appels met appels).

# Afhankelijk of png of pdf gekozen wordt als datatype wordt alles opgeslaan in plots_compare-png of plots_compare-pdf.

# Als titel van grafiek en naam van bestand wordt de naam van de folder gebruikt waardat de te vergelijken bestanden in zitten
# deze folder moet dus aangemaakt worden in de folder datalogger_files.

# De code werkt enkel in de gekozen folder, om de folder aan te passen moet de variabele: compareFolder aangepast worden hieronder.
##filetype kiezen: 'png' of 'pdf'
filetype='png'                                          # 'pdf' is betere qualiteit
compareFolder='calibratie 1 vs calibratie 2'                         # kies hier folder met de te vergelijken metingen in

## bekijk waar data begint en maak vanaf die rij een dataframe aan
def readData(csvfile):
    df = pd.read_csv(csvfile,usecols = [0],encoding='UTF-16 LE')
    rowsToSkip=df.loc[df[df.columns[0]] == 'Scan'].index[0]+1
    df = pd.read_csv(csvfile,skiprows=rowsToSkip,encoding='UTF-16 LE')
    return df

## functie die de alarm colommen verwijderd uit de df
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
    df.insert(2, 'Time_seconds', None)                  # voeg colomn looptijd in seconden toe
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

## gaat over alle csvfiles in de /datalogger_files map en maakt een dictionairy aan met alle dataframes van die csvfiles er in 
def dataframeDict():
    dfs = {}                                            # Create an empty dictionary to hold the DataFrames
    folder_path = f'datalogger_files/{compareFolder}'   # hier zitten alle csv files in
    try:
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')] 
    except:
        print('ERROR: gekozen comparison folder bestaat niet of staat niet op juiste plaats (variabele: compareFolder)')
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = readData(file_path)                        #dataframe uitlezen
        df = dropAlarmCols(df)                          #enkele bewerkingen op dataframe
        df = formatTimecol(df)
        df = avgColumn(df)
        print(df.head(50))
        dfs[file] = df                                  # Use the file name as the key and store the DataFrame in the dictionary
    return dfs

## plot van alle dataframes, de gekozen thermocouples(kiezen bij volgende functie)
def plotMeasurement(filetype):
    fig, ax = plt.subplots(1)
    ax.grid(True)
    dfs=dataframeDict()    
    plt.xlabel("Time [s]")                              # as tittels toevoegen
    plt.ylabel("Temperature [K]")
    # ax.set_ylim(ymin=15)                              # startwaarde op y-as instellen (optioneel)
    for file, df in dfs.items():
        file=file.split('-')                            # text tussen twee '-' nemen, anders volledige naam van file als titel en filename van grafiek
        if len(file) >= 3:
            placeData(file[1],df)
        else:
            placeData(file[0][:-4],df)
    plt.legend()
    plt.title(compareFolder)
    plt.savefig(f'plots_compare-{filetype}/{compareFolder}.{filetype}')       

def placeData(fileName,df):
    try:
        plt.plot(df['Time_seconds'], df.iloc[:, [2]],label=f'{fileName}-{df.columns[2]}')       #thermocouples kiezen
        plt.plot(df['Time_seconds'], df.iloc[:, [4]],label=f'{fileName}-{df.columns[4]}')
        plt.plot(df['Time_seconds'], df.iloc[:, [6]],label=f'{fileName}-{df.columns[6]}')
    except:
        print('ERROR: gekozen columns zitten niet in dataFrame')



plotMeasurement(filetype)



