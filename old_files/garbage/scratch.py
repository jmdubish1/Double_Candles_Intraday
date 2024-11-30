import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def concat_excel_files(folder_path, sheet_name):
    # Get a list of all .xlsx files in the folder
    excel_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
    all_data = []

    # Relevant column names
    relevant_columns = ["Algo_PnL", "Pred_PnL", "Two_Dir_Pred_PnL"]

    # Read and concatenate data from the specified sheet
    for file in excel_files:
        file_path = os.path.join(folder_path, file)
        try:
            # Read the specific sheet from the Excel file
            df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
            df = df.dropna(subset=['DateTime']).reset_index(drop=True)
            df = df.drop(columns='Unnamed: 0')
            all_data.append(df)

        except Exception as e:
            print(f"Error reading {sheet_name} from {file}: {e}")

    # Concatenate all DataFrames
    if all_data:
        all_data = pd.concat(all_data, ignore_index=True)

        # Convert to numeric, handling errors if any column types are incorrect
        all_data.iloc[:, 1:] = all_data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

        # Return the combined DataFrame
        return all_data
    else:
        print(f"No valid data found for sheet {sheet_name}.")
        return None


def plot_rolling_sum(df, sheet_name):
    # Calculate the rolling sum if the DataFrame is not empty
    if df is not None and not df.empty:
        # Calculate rolling sum
        rolling_df = df.copy()
        rolling_df['DateTime'] = pd.to_datetime(rolling_df['DateTime'], errors='coerce')
        rolling_df = rolling_df.sort_values(by='DateTime').reset_index(drop=True)
        rolling_df['Algo_PnL_total'] = df['Algo_PnL'].cumsum()
        rolling_df['Pred_PnL_total'] = df['Pred_PnL'].cumsum()
        rolling_df['Two_Dir_Pred_PnL_total'] = df['Two_Dir_Pred_PnL'].cumsum()

        plt.figure(figsize=(12, 7))
        plt.plot(rolling_df['DateTime'], rolling_df["Algo_PnL_total"], label="Algo_PnL (Rolling Sum)")
        plt.plot(rolling_df['DateTime'], rolling_df["Pred_PnL_total"], label="Pred_PnL (Rolling Sum)")
        plt.plot(rolling_df['DateTime'], rolling_df["Two_Dir_Pred_PnL_total"], label="Two_Dir_Pred_PnL (Rolling Sum)")
        plt.gcf().autofmt_xdate()

        rolling_df.to_excel(f'{sheet_name}_total.xlsx')

        plt.xlabel("Index")
        plt.ylabel("Rolling Sum Values")
        plt.title(f"{sheet_name}")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print(f"No data available to plot for {sheet_name}.")

side = 'Bull'
folder_main = r'C:\Users\jmdub\Documents\Trading\Futures\Strategy Info\Double_Candles\ATR\NQ\5min\5min_test_20years'
# Usage
folder_path = f'{folder_main}\\{side}\\Data\\{side}_2'
# folder_path = r'C:\Users\jmdub\Documents\Trading\Futures\Strategy Info\Double_Candles\ATR\NQ\15min\15min_test_20years\good candidates\Bull_2\Data\Bull_129'
bull_wl_data = concat_excel_files(folder_path, f"{side}_WL")
bull_pnl_data = concat_excel_files(folder_path, f"{side}_PnL")

# rolling_df = pd.DataFrame()
# rolling_df['Algo_PnL'] = bull_wl_data['Algo_PnL'].cumsum()
# rolling_df['Pred_PnL'] = bull_wl_data['Pred_PnL'].cumsum()
# rolling_df['Two_Dir_Pred_PnL'] = bull_wl_data['Two_Dir_Pred_PnL'].cumsum()

# Plot the graphs
plot_rolling_sum(bull_wl_data, f"{side}_WL")
plot_rolling_sum(bull_pnl_data, f"{side}_PnL")



