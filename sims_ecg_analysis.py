import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Function to plot histograms
def plot_histogram(data, title, xlabel, ylabel, rotation=0, colors=None):
    plt.figure(figsize=(6, 4))
    if colors is None:
        data.plot(kind='bar')
    else:
        data.plot(kind='bar', color=colors)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.show()


# ##############################################
# #Read in files
# ##############################################
result_dir = '/Users/morganfitzgerald/Projects/ecg_param/sims/docs/timedomain_results/'


# ##############################################
# #R squared to assess model fit 
# ##############################################


# Get a list of result files ending with '_ecg_output.csv'
result_files = [f for f in os.listdir(result_dir) if f.endswith('_ecg_output.csv')]

# Initialize a dictionary to store R-squared values
r_squared_values = {}

# Iterate over each result file
for file_name in result_files:
    # Extract the noise level from the file name
    noise_level = file_name.split('_')[0].zfill(4)
    # Read the CSV file into a DataFrame
    df = pd.read_csv(os.path.join(result_dir, file_name))
    # Drop NaN values and filter out R-squared values less than or equal to 0
    r_squared_vals = df['r_squared'].dropna().loc[df['r_squared'] > 0]
    # Store the filtered R-squared values in the dictionary
    r_squared_values[noise_level] = r_squared_vals

# Plotting the distribution  R^2 values
plt.figure(figsize=(8, 6))
for noise_level, r_squared_vals in r_squared_values.items():
    plt.hist(r_squared_vals, bins=20, alpha=0.5, label=f'Noise Level: {noise_level}')

plt.title('Distribution of $R^2$ Values')
plt.xlabel('$R^2$ Value')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()

# # Calculating and printing the overall average R^2 value
# overall_average_r_squared = full_df['Average_R_squared'].mean()
# print(f"The overall average R^2 value across all participants is: {overall_average_r_squared:.4f}")

# # Process and plot R^2 values
# r_squared_values = {}
# result_files = [f for f in os.listdir(result_dir) if f.endswith('_ecg_output.csv')]

# for file_name in result_files:
#     subject_num = file_name.split('_')[0].zfill(4)
#     df = pd.read_csv(os.path.join(result_dir, file_name))
#     r_squared_non_zero = df['r_squared'].dropna()[df['r_squared'] > 0]
#     r_squared_values[subject_num] = r_squared_non_zero.mean()

# average_r_squared_df = pd.DataFrame(list(r_squared_values.items()), columns=['ID', 'Average_R_squared'])
# full_df = pd.merge(subject_info_df, average_r_squared_df, on='ID')

# # Plot R^2 values by Age Range and Sex
# def plot_grouped_bar(data, group, title, xlabel, ylabel):
#     plt.figure(figsize=(4, 4))
#     grouped_data = data.groupby(group)['Average_R_squared'].mean()
#     grouped_data.plot(kind='bar', color='skyblue')
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.xticks(rotation=45 if group == 'Age_range' else 0)
#     plt.tight_layout()
#     plt.show()

# plot_grouped_bar(full_df, 'Age_range', 'Average $R^2$ Values by Age Range', 'Age Range', 'Average $R^2$ Value')
# plot_grouped_bar(full_df, 'Sex_class', 'Average $R^2$ Values by Sex', 'Sex', 'Average $R^2$ Value')



# # ##############################################
# # #Sharpness deriv
# # ##############################################

# # Define the result directory
# timed_result_dir = '../docs/saved_files/timedomain_results/'

# # List of peaks and their corresponding sharpness deriv columns
# peaks = ['p', 'q', 'r', 's', 't']
# sharpness_deriv_cols = {peak: f'sharpness_deriv_{peak}' for peak in peaks}

# # Initialize a dictionary to store sharpness deriv values for each peak
# sharpness_deriv_values = {peak: [] for peak in peaks}

# # Get the list of result files
# result_files = [f for f in os.listdir(timed_result_dir) if f.endswith('_ecg_output.csv')]

# # Loop through each result file and extract sharpness deriv values
# for file_name in result_files:
#     df = pd.read_csv(os.path.join(timed_result_dir, file_name))
#     for peak, col in sharpness_deriv_cols.items():
#         # Extract sharpness deriv values and add them to the corresponding list
#         sharpness_deriv_values[peak].extend(df[col].dropna())

# # Function to plot histograms with axis limits based on data range
# def plot_histogram(data, title, xlabel, ylabel, bins=20, color='skyblue', extra_padding=0.1):
#     plt.figure(figsize=(6, 4))
#     plt.hist(data, bins=bins, color=color, edgecolor='black')
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)

#     # Calculate axis limits based on min and max data values with extra padding
#     min_val = min(data)
#     max_val = max(data)

#     # Extra padding to ensure the plot doesn't touch the edges
#     xlim = (min_val - extra_padding * abs(min_val), max_val + extra_padding * abs(max_val))

#     # Set the x-axis limits
#     plt.xlim(xlim)

#     plt.tight_layout()
#     plt.show()

# # Plot histograms for each peak, using axis limits based on data range
# for peak in peaks:
#     title = f'Histogram of Sharpness deriv ({peak.upper()})'
#     xlabel = f'Sharpness deriv ({peak.upper()})'
#     ylabel = 'Frequency'

#     plot_histogram(sharpness_deriv_values[peak], title, xlabel, ylabel)



# # Function to plot box plots for given data
# def plot_box_plot(data, title, xlabel, ylabel):
#     plt.figure(figsize=(8, 6))
#     plt.boxplot(data, patch_artist=True, showfliers=True)  # Show outliers as dots
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.tight_layout()
#     plt.show()

# # Create a list of sharpness deriv values for all peaks
# deriv_peaks_data = [sharpness_deriv_values[peak] for peak in peaks]

# # Plot box plots for sharpness deriv across all peaks
# plot_box_plot(deriv_peaks_data, 'Box Plot of Sharpness Deriv for Each Peak', 'Peaks', 'Sharpness Deriv')

# # You can customize the x-axis labels to reflect the peaks
# plt.xticks(ticks=range(1, len(peaks) + 1), labels=[peak.upper() for peak in peaks], rotation=45)

# ##############################################
# #Average Exponent and Offset Values By Noise 
# ##############################################
# # Load and merge datasets
# subject_details = pd.read_csv('../docs/saved_files/spectral_results/subject_details.csv')
# sp_results = pd.read_csv('../docs/saved_files/spectral_results/sp_results.csv')
# merged_data = pd.merge(subject_details, sp_results, on='Subject')
# merged_data = merged_data[merged_data['Age_range'] != 'Unknown']

# # Calculate mean values for each parameter by Age_range
# mean_values = merged_data.groupby('Age_range')[['Offset_sp', 'Exponent_sp']].mean().reset_index()


# def plot_avg_values_histogram_direct(data, parameter, title):
#     # Sort data
#     sorted_data = data.sort_values(by='Age_range')
#     # Create figure and axis
#     fig, ax = plt.subplots(figsize=(12, 6))
#     # Plot directly using Matplotlib
#     ax.bar(sorted_data['Age_range'], sorted_data[parameter], color='skyblue')
#     # Set title and labels
#     ax.set_title(title)
#     ax.set_xlabel('Age Range')
#     ax.set_ylabel('Average Value')
#     # Set x-ticks
#     ax.set_xticks(range(len(sorted_data['Age_range'])))
#     ax.set_xticklabels(sorted_data['Age_range'], rotation=45)
#     # Apply tight layout and show plot
#     plt.tight_layout()
#     plt.show()

# ##Trying weighting to adjust for sample size differnces across age
# count_data = merged_data.groupby('Age_range')['Subject'].count().reset_index().rename(columns={'Subject': 'Count'})
# mean_values = mean_values.merge(count_data, on='Age_range')

# # Calculate weighted averages
# mean_values['Weighted_Offset_sp'] = mean_values['Offset_sp'] * mean_values['Count'] / mean_values['Count'].sum()
# mean_values['Weighted_Exponent_sp'] = mean_values['Exponent_sp'] * mean_values['Count'] / mean_values['Count'].sum()

# # # Plot histograms for weighted average Offset and Exponent values
# # plot_avg_values_histogram_direct(mean_values, 'Weighted_Offset_sp', 'Weighted Average Offset Values by Age Group')
# # plot_avg_values_histogram_direct(mean_values, 'Weighted_Exponent_sp', 'Weighted Average Exponent Values by Age Group')


# ##Weighted average significance testing: Use Welch's t-test due to uneven groups!
# from scipy.stats import ttest_ind

# # Age groups definition based on 'Age_range'
# age_20_to_30_ranges = ['20-24 years', '25-29 years']
# age_over_50_ranges = ['50-54 years', '55-59 years', '60-64 years', '65-69 years', '70-74 years', '75-79 years', '80-84 years', '85-92 years']

# # Separating the merged_data into two new groups
# age_20_to_30 = merged_data[merged_data['Age_range'].isin(age_20_to_30_ranges)]
# age_over_50 = merged_data[merged_data['Age_range'].isin(age_over_50_ranges)]

# # Ensure you've imported the necessary function for the t-test
# from scipy.stats import ttest_ind

# # Perform Independent Samples t-test for Exponent_sp between the two age groups
# t_stat_exp_20_30_vs_50, p_value_exp_20_30_vs_50 = ttest_ind(age_20_to_30['Exponent_sp'], age_over_50['Exponent_sp'], equal_var=False)  # Assuming unequal variances

# # Perform Independent Samples t-test for Offset_sp between the two age groups
# t_stat_offset_20_30_vs_50, p_value_offset_20_30_vs_50 = ttest_ind(age_20_to_30['Offset_sp'], age_over_50['Offset_sp'], equal_var=False)  # Assuming unequal variances

# # Output the results
# print(f'Exponent_sp t-test (20-30 vs. >50): t={t_stat_exp_20_30_vs_50}, p={p_value_exp_20_30_vs_50}')
# print(f'Offset_sp t-test (20-30 vs. >50): t={t_stat_offset_20_30_vs_50}, p={p_value_offset_20_30_vs_50}')


# #checkign 20-30 adn 55+

