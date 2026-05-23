import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv


load_dotenv()
DB_URL = os.getenv("DB_URL")

if not DB_URL:
   
    exit()


engine = create_engine(DB_URL)


csv_file_path = 'fps_benchmark.csv'

df = pd.read_csv(csv_file_path)



cpus_df = df[['CpuName', 'CpuNumberOfCores', 'CpuNumberOfThreads', 'CpuFrequency', 'CpuTurboClock', 'CpuTDP']].copy()
cpus_df = cpus_df.drop_duplicates(subset=['CpuName']) 

cpus_df.rename(columns={
    'CpuNumberOfCores': 'CpuCores',
    'CpuNumberOfThreads': 'CpuThreads',
    'CpuFrequency': 'CpuBaseClock',
    'CpuTurboClock': 'CpuBoostClock'
}, inplace=True)



gpus_df = df[['GpuName', 'GpuMemorySize', 'GpuBandwidth', 'GpuBoostClock']].copy()
gpus_df = gpus_df.drop_duplicates(subset=['GpuName'])

gpus_df.rename(columns={
    'GpuMemorySize': 'GpuVRAM'
}, inplace=True)

gpus_df['GpuTDP'] = 250.0 



games_df = df[['GameName']].copy().drop_duplicates()



settings_df = df[['GameSetting']].copy().drop_duplicates()
settings_df.rename(columns={'GameSetting': 'SettingName'}, inplace=True)

cpus_df.to_sql('cpus', engine, if_exists='replace', index=False)
gpus_df.to_sql('gpus', engine, if_exists='replace', index=False)
games_df.to_sql('games', engine, if_exists='replace', index=False)
settings_df.to_sql('game_settings', engine, if_exists='replace', index=False)

