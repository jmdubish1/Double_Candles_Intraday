import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def concat_excel_files(f_path, sheet_name):
    # Get a list of all .xlsx files in the folder
    excel_files = [f for f in os.listdir(f_path) if f.endswith('.xlsx')]
    all_data = []

    # Read and concatenate data from the specified sheet
    for file in excel_files:
        file_path = os.path.join(f_path, file)
        try:
            # Read the specific sheet from the Excel file
            df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
            df = df.dropna(subset=['DateTime']).reset_index(drop=True)
            df = df.drop(columns='Unnamed: 0', errors='ignore')  # Avoid errors if the column doesn't exist
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {sheet_name} from {file}: {e}")

    # Concatenate all DataFrames
    if all_data:
        all_data = pd.concat(all_data, ignore_index=True)
        # Convert to numeric, handling errors if any column types are incorrect
        all_data.iloc[:, 1:] = all_data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
        return all_data
    else:
        print(f"No valid data found for sheet {sheet_name}.")
        return None


def plot_rolling_sum(df, sheet_name, param_id, f_path):
    if df is not None and not df.empty:
        # Calculate rolling sum
        rolling_df = df.copy()
        rolling_df['DateTime'] = pd.to_datetime(rolling_df['DateTime'], errors='coerce')
        rolling_df = rolling_df.sort_values(by='DateTime').reset_index(drop=True)
        rolling_df['Algo_PnL_total'] = df['Algo_PnL'].cumsum()
        rolling_df['Pred_PnL_total'] = df['Pred_PnL'].cumsum()
        rolling_df['Two_Dir_PnL_total'] = df['Two_Dir_Pred_PnL'].cumsum()

        plt.figure(figsize=(12, 7))
        plt.plot(rolling_df['DateTime'], rolling_df["Algo_PnL_total"], label="Algo_PnL (Rolling Sum)")
        plt.plot(rolling_df['DateTime'], rolling_df["Pred_PnL_total"], label="Pred_PnL (Rolling Sum)")
        plt.plot(rolling_df['DateTime'], rolling_df["Two_Dir_PnL_total"], label="Two_Dir_Pred_PnL (Rolling Sum)")
        plt.gcf().autofmt_xdate()

        rolling_df.to_excel(f'{sheet_name}_total.xlsx', index=False)

        plt.xlabel("Index")
        plt.ylabel("Rolling Sum Values")
        plt.title(f"{sheet_name}_{param_id}")
        plt.legend()
        plt.grid(True)
        # plt.show()

        img_loc = os.path.join(f_path, f"{param_id}_{sheet_name}.png")
        plt.savefig(img_loc)

        plt.close()

    else:
        print(f"No data available to plot for {sheet_name}.")


side = 'Bull'
folder_main = r'C:\Users\jmdub\Documents\Trading\Futures\Strategy Info\Double_Candles\ATR\NQ\15min\15min_test_20years'
folder_path = os.path.join(folder_main, side, "Data")

# Filter only directories
result_files = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
for param in result_files:
    print(f'param: {param}')
    rf = os.path.join(folder_path, param)

    bull_wl_data = concat_excel_files(rf, f"{side}_WL")
    bull_pnl_data = concat_excel_files(rf, f"{side}_PnL")

    # Plot the graphs
    plot_rolling_sum(bull_wl_data, f"{side}_WL", param, folder_path)
    plot_rolling_sum(bull_pnl_data, f"{side}_PnL", param, folder_path)

