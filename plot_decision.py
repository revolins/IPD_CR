import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import argparse
import glob
import re
from tqdm import tqdm
from statsmodels.tsa.stattools import grangercausalitytests

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) #For those pesky deprecation warnings
# https://www.statsmodels.org/dev/generated/statsmodels.stats.stattools.durbin_watson.html
def join_path(output_folder, filename):
        return os.path.join(output_folder, filename)

def extract_floats(s):
    pattern = r"[-+]?\d*\.\d+"
    floats = re.findall(pattern, s)
    return [float(num) for num in floats]


def plot_noisy(args):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dll_list, bits_list = [], []
    for i, folder_name in enumerate(os.listdir(dir_path)):
        if args.env_test in folder_name and 'stat_output' not in folder_name and 'coop' not in folder_name and 'hostile' not in folder_name:
            assert 'noise' in folder_name, f'noise not found in name of folder - {folder_name}'
            df = pd.read_csv(join_path(folder_name, 'decision_list_length_overtime.csv')) # Which came first the dataframe or the dataframe?
            bits_df = pd.read_csv(join_path(folder_name, 'all_bits_df_Summary_comp_more_values.csv'))
            nums = extract_floats(folder_name)
            print(nums)
            bits_df['Condition'] = nums[0]
            df['Condition'] = nums[0]
            dll_list.append(df)
            bits_list.append(bits_df)
    organism_columns = [col for col in dll_list[0].columns if 'Organism' in col]
    bit_columns = [col for col in bits_list[0].columns if 'Organism' in col]
    dll_df = pd.concat(dll_list, ignore_index=True)
    tot_bit_df = pd.concat(bits_list, ignore_index=True)
    
    dll_df['Mean'] = dll_df.apply(lambda row: np.average(row[:len(organism_columns)]), axis=1)
    weights = np.arange(1, len(bit_columns) + 1)
    weights = weights / weights.sum()
    tot_bit_df['Mean'] = tot_bit_df.apply(lambda row: np.average(row[:len(bit_columns)], weights=weights), axis=1)
    
    summary_df = dll_df.groupby(['Condition', 'Generation'], observed=True).agg(
        group_mean=('Mean', 'mean'),
        group_sd=('Mean', 'std')
    ).reset_index()

    conditions = summary_df['Condition'].unique()
    summary_df.to_csv('debug_dll.csv', header=True)
    palette = sns.color_palette("husl", len(conditions))
    print(summary_df)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=summary_df, x='Generation', y='group_mean', hue='Condition', style='Condition', markers=False, dashes=False, err_style="band", errorbar='ci', palette=palette)
    plt.title(f'Average Decision List Length per Generation')
    plt.ylabel(f'Decision List Length')
    plt.grid(False)
    plt.savefig(f'Average_Decision_List_Length_Overtime_{args.env_test}.png')

    
    plot_bit_df = tot_bit_df.groupby(['Condition', 'Generation'], observed=True).agg(
        group_mean=('Mean', 'mean'),
        group_sd=('Mean', 'std')
    ).reset_index()
    # plot_bit_df.loc[plot_bit_df['group_sd'] == 0.0] = np.nan
    # plot_bit_df.loc[plot_bit_df['group_sd'] == np.min(plot_bit_df['group_sd'])] = np.nan
    # plot_bit_df.ffill(inplace=True)

    conditions = plot_bit_df['Condition'].unique()
    plot_bit_df.to_csv('debug_mem.csv', header=True)
    palette = sns.color_palette("husl", len(conditions))
    print(plot_bit_df)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=plot_bit_df, x='Generation', y='group_mean', hue='Condition', style='Condition', markers=False, dashes=False, err_style="band", errorbar='ci', palette=palette)
    plt.title(f'Average Bits Of Summary per Generation')
    plt.ylabel(f'Average Bits Of Summary')
    plt.grid(False)
    plt.savefig(f'Average_Bits_Of_Summary_Overtime_{args.env_test}.png')

    forward_df = pd.concat([plot_bit_df['group_mean'], summary_df['group_mean']], axis=1)
    inv_df = pd.concat([summary_df['group_mean'], plot_bit_df['group_mean']], axis=1)
    with open(f"{args.env_test}_stat_output.txt", "a") as f:
        f.write("########################################################################################################")
        f.write("#####################################        Correlation       #########################################")
        f.write("#########################       Total Memory versus Decision List Length         #######################")
        f.write("########################################################################################################")
        try:
            pearson_corr = summary_df['group_mean'].corr(plot_bit_df['group_mean'], method='pearson')
            f.writelines(f'{args.env_test} - Pearson Correlation test results: {pearson_corr}')
            f.write('\n')
        except:
            f.write(f"{args.env_test} - Pearson Correlation test failed, column mismatch likely")
            f.write('\n')
            print(f"{args.env_test} - Pearson Correlation test failed, column mismatch likely", flush=True)
        try:
            pearson_inv = plot_bit_df['group_mean'].corr(summary_df['group_mean'], method='pearson')
            f.writelines(f'{args.env_test} - Pearson Inverse test results: {pearson_inv}')
            f.write('\n')
        except:
            f.write(f"{args.env_test} - Pearson Inverse test failed, column mismatch likely")
            f.write('\n')
            print(f"{args.env_test} - Pearson Inverse test failed, column mismatch likely", flush=True)

        try:
            spearman_corr = summary_df['group_mean'].corr(plot_bit_df['group_mean'], method='spearman')
            f.writelines(f'{args.env_test} - Spearman Correlation test results: {spearman_corr}')
            f.write('\n')
        except:
            f.write(f"{args.env_test} - Spearman Correlation test failed, column mismatch likely")
            f.write('\n')
            print(f"{args.env_test} - Spearman Correlation test failed, column mismatch likely", flush=True)
        
        try:
            spearman_inv = plot_bit_df['group_mean'].corr(summary_df['group_mean'], method='spearman')
            f.writelines(f'{args.env_test} - Spearman Inverse test results: {spearman_inv}')
            f.write('\n')
        except:
            f.write(f"{args.env_test} - Spearman Inverse test failed, column mismatch likely")
            f.write('\n')
            print(f"{args.env_test} - Spearman Inverse test failed, column mismatch likely", flush=True)
        try:
            granger_caus = grangercausalitytests(forward_df, maxlag=2, verbose=True)
            f.writelines(f'{args.env_test} - Granger Causality test results: {granger_caus}')
            f.write('\n')
        except:
            f.write(f"{args.env_test} - Granger Causality test failed, column mismatch likely")
            f.write('\n')
            print(f"{args.env_test} - Granger Causality test failed, column mismatch likely", flush=True)
        try:
            granger_inv = grangercausalitytests(inv_df, maxlag=2, verbose=True)
            f.writelines(f'{args.env_test} - Granger Inverse test results: {granger_inv}')
            f.write('\n')
        except:
            f.write(f"{args.env_test} - Granger Inverse test failed, column mismatch likely")
            f.write('\n')
            print(f"{args.env_test} - Granger Inverse test failed, column mismatch likely", flush=True)

def main():
    arg_parser = argparse.ArgumentParser(
        description='Plotting function for handling noisy_data folder.')
    
    # Expects 1 argument: output folder
    arg_parser.add_argument("-i", "--input_folder", type=str, default="temp_test")
    arg_parser.add_argument("-et", "--env_test", type=str, default="coop", help="(str) (DEFAULT = 'coop'), choose between hostile and coop")
    args = arg_parser.parse_args()
    plot_noisy(args)
    

if __name__ == "__main__":
    main()