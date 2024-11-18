#!/usr/bin/env python
# coding: utf-8

# ## Variable Renewable Energy (VRE) assessment and forecast
#
# ### Project objectives
# <div class="alert alert-block alert-info">
#
# - Assess the onshore wind or solar photovoltaic hourly production over in metropolitan France regions using climate data and capacity factor observations.
# - Predict the VRE power ahead of time.
# </div>

# ### Dataset
#
# - Observed monthly VRE capacity factors averaged over metropolitan France regions from 2014 to 2021
# - Climate variables of your choice from a global reanalysis with an hourly sampling from 2010 to 2019

# ### First steps
#
# - Choose from solar or wind power
# - Read about solar/wind production assessment and forecast
# - Estimate the hourly solar/wind production

# ### Reading the data

from pathlib import Path
import matplotlib
matplotlib.use('Agg') #Non-interactive backend for Xcode
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import numpy as np

# Directories where you saved the data
data_dir_energy = Path("/Users/afonsobarroco/Desktop/Academic:Professional/24:25/1st Term/Machine Learning for Climate and Energy/Project/energy_france")
data_dir_climate = Path("/Users/afonsobarroco/Desktop/Academic:Professional/24:25/1st Term/Machine Learning for Climate and Energy/Project/climate_france_2019")
                        
# Template filenames
filename_mask = 'mask_datagouv_french_regions_merra2_Nx_France.nc'
filename_climate = 'merra2_area_selection_output_{}_merra2_2019-2019.nc'
filename_energy = 'reseaux_energies_{}.csv'

# Set keyword arguments for pd.read_csv
kwargs_read_csv = dict(index_col=0, header=0, parse_dates=True)

# Read and plot grid point-region mask
filepath_mask = Path(data_dir_climate, filename_mask)
ds_mask = xr.load_dataset(filepath_mask)
da_mask = ds_mask['mask']
plt.figure()
plt.title('Grid Point Region Mask')
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.scatter(da_mask['lon'], da_mask['lat'], c=da_mask, cmap='Set1')
plt.colorbar(label='Region')

# Read a climate variable and plot its mean over time
variable_name = 'surface_downward_radiation'
filename = filename_climate.format(variable_name)
filepath = Path(data_dir_climate, filename)
da_climate = xr.load_dataset(filepath)[variable_name]
plt.figure()
plt.title('Surface Downward Radiation over Time')
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.scatter(da_mask['lon'], da_mask['lat'], c=da_climate.mean('time'))
sc_climate = plt.scatter(da_mask['lon'], da_mask['lat'], c=da_climate.mean('time'))
plt.colorbar(sc_climate, label='Mean Climate Variable')

# Compute regional mean of climate variable
da_climate_reg = da_climate.groupby(da_mask).mean().rename(mask='region')
da_climate_reg['region'] = ds_mask['region'].values

# Plot time series for the regional mean climate variable with 'region' as hue
plt.figure()
da_climate_reg.plot.line(x='time', hue='region')
plt.xlabel('Time')
plt.ylabel('Regional Mean of Climate Variable')
plt.title('Regional Mean of Climate Variable over Time')
plt.show()


# Plot energy variable time series
variable_name = 'capacityfactor_pv'
filename = filename_energy.format(variable_name)
filepath = Path(data_dir_energy, filename)
df_energy = pd.read_csv(filepath, index_col=0, header=0, parse_dates=True)
plt.figure()
df_energy.plot()
plt.title('PV Energy Mean Over Time')
plt.xlabel('Time')
plt.ylabel('Energy Variable in MWh')

plt.show()

# Load the surface_downward_radiation NetCDF file and inspect it for any unexpected zero values
file_path = Path(data_dir_climate, filename_climate.format('surface_downward_radiation'))
da_climate = xr.load_dataset(file_path)['surface_downward_radiation']

# Checking if the entire dataset is zero
if (da_climate == 0).all():
    print("Warning: The 'surface_downward_radiation' dataset contains only zeros. Please verify the data source.")
else:
    print("The 'surface_downward_radiation' dataset contains non-zero values. Proceeding with analysis.")
    
# Display dataset information for debugging
print(da_climate)
da_climate.mean(dim='time').plot()  # Mean over time for visualization
plt.show()

import matplotlib.pyplot as plt

# Plotting the 'surface_downward_radiation' variable over time, assuming `df_combined` has time-indexed data for this variable
plt.figure(figsize=(10, 6))
df_combined['surface_downward_radiation'].plot()
plt.title('Surface Downward Radiation Over Time')
plt.xlabel('Time')
plt.ylabel('Surface Downward Radiation')
plt.grid(True)
plt.show()

# Read a climate variable and plot its mean over time
variable_name = 'surface_density'
filename = filename_climate.format(variable_name)
filepath = Path(data_dir_climate, filename)
da_climate = xr.load_dataset(filepath)[variable_name]
plt.figure()
plt.title('Surface Density over Time')
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.scatter(da_mask['lon'], da_mask['lat'], c=da_climate.mean('time'))
sc_climate = plt.scatter(da_mask['lon'], da_mask['lat'], c=da_climate.mean('time'))
plt.colorbar(sc_climate, label='surface density')

# Compute regional mean of climate variable
da_climate_reg = da_climate.groupby(da_mask).mean().rename(mask='region')
da_climate_reg['region'] = ds_mask['region'].values

# Plot time series for the regional mean climate variable with 'region' as hue
plt.figure()
da_climate_reg.plot.line(x='time', hue='region')
plt.xlabel('Time')
plt.ylabel('Regional Mean of surface density')
plt.title('Regional Mean of surface density over Time')
plt.show()

import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression

# Load PV energy data
df_energy = pd.read_csv('data/energy_france/reseaux_energies_capacityfactor_pv.csv', index_col=0, parse_dates=True)

# Load climate data for each variable
data_files = {
    'surface_downward_radiation': 'data/climate_france_2019/merra2_area_selection_output_surface_downward_radiation_merra2_2019-2019.nc',
    'surface_temperature': 'data/climate_france_2019/merra2_area_selection_output_surface_temperature_merra2_2019-2019.nc',
    'surface_specific_humidity': 'data/climate_france_2019/merra2_area_selection_output_surface_specific_humidity_merra2_2019-2019.nc',
    'surface_density': 'data/climate_france_2019/merra2_area_selection_output_surface_density_merra2_2019-2019.nc',
    'height_500': 'data/climate_france_2019/merra2_area_selection_output_height_500_merra2_2019-2019.nc',
    'meridional_wind': 'data/climate_france_2019/merra2_area_selection_output_meridional_wind_merra2_2019-2019.nc',
    'upper_meridional_wind': 'data/climate_france_2019/merra2_area_selection_output_upper_meridional_wind_merra2_2019-2019.nc',
    'upper_zonal_wind': 'data/climate_france_2019/merra2_area_selection_output_upper_zonal_wind_merra2_2019-2019.nc',
    'zonal_wind': 'data/climate_france_2019/merra2_area_selection_output_zonal_wind_merra2_2019-2019.nc'
}


# Read and process each climate variable to calculate its regional mean over time
climate_datasets = {}
for var, filepath in data_files.items():
    ds = xr.open_dataset(filepath)
    da_var = ds[var]
    # Assuming the mask region logic is pre-defined, e.g., in a region mask
    da_var_regional_mean = da_var.mean(dim=['stacked_dim'])  # Simplified; change as per grid
    climate_datasets[var] = da_var_regional_mean.to_dataframe(name=var)

# Combine all climate variables and energy data
df_climate = pd.concat(climate_datasets.values(), axis=1)
df_combined = pd.concat([df_energy, df_climate], axis=1).dropna()

# Display the first few rows of the combined dataset
df_combined.head()

df_combined.tail()

import xarray as xr

# Define a function to check for NaN values in a NetCDF file
def check_nan_in_nc(filepath, variable_name):
    # Open the NetCDF file
    ds = xr.open_dataset(filepath)
    
    # Check if the variable exists in the dataset
    if variable_name in ds:
        da_var = ds[variable_name]
        
        # Check for NaN values using isnull().sum() to count the NaNs
        nan_count = da_var.isnull().sum().item()
        
        if nan_count > 0:
            print(f"Variable '{variable_name}' in '{filepath}' contains {nan_count} NaN values.")
        else:
            print(f"Variable '{variable_name}' in '{filepath}' contains no NaN values.")
    else:
        print(f"Variable '{variable_name}' not found in '{filepath}'.")

# Example usage:
# Specify the path to the NetCDF file and the variable you want to check
file_path = "data/climate_france_2019/merra2_area_selection_output_surface_downward_radiation_merra2_2019-2019.nc"
variable = "surface_downward_radiation"  # Replace with the correct variable name

# Run the NaN check function
check_nan_in_nc(file_path, variable)

# Count the number of zero values in the 'surface_downward_radiation' column
zero_count = (df_combined['surface_downward_radiation'] == 0).sum()
print(f"The number of zeros in 'surface_downward_radiation' column is: {zero_count}")

import matplotlib.pyplot as plt
import numpy as np

# List of climate variables and regions
climate_variables = [
    'surface_downward_radiation', 'surface_temperature',
    'surface_specific_humidity', 'surface_density',
    'zonal_wind', 'height_500', 'upper_zonal_wind',
    'meridional_wind', 'upper_meridional_wind'
]

regions = [
    'Auvergne-Rhône-Alpes', 'Bourgogne-Franche-Comté', 'Bretagne',
    'Centre-Val de Loire', 'Grand Est', 'Hauts-de-France',
    'Île-de-France', 'Normandie', 'Nouvelle-Aquitaine', 'Occitanie',
    'Pays de la Loire', 'Provence-Alpes-Côte d\'Azur'
]

# Generate scatter plots for each climate variable and region combination
for climate_var in climate_variables:
    for region in regions:
        # Calculate correlation coefficient
        correlation = np.corrcoef(df_combined[climate_var], df_combined[region])[0, 1]
        
        # Create scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(df_combined[climate_var], df_combined[region], alpha=0.5)
        plt.xlabel(climate_var)
        plt.ylabel(f'Capacity Factor - {region}')
        plt.title(f"{climate_var} vs Capacity Factor ({region})\nCorrelation: {correlation:.2f}")
        plt.grid(True)
        plt.show()


import seaborn as sns  # Using seaborn for an easier heatmap visualization

# Define the list of regions and climate variables
regions = [
    'Auvergne-Rhône-Alpes', 'Bourgogne-Franche-Comté', 'Bretagne',
    'Centre-Val de Loire', 'Grand Est', 'Hauts-de-France',
    'Île-de-France', 'Normandie', 'Nouvelle-Aquitaine', 'Occitanie',
    'Pays de la Loire', 'Provence-Alpes-Côte d\'Azur'
]

climate_variables = [
    'surface_downward_radiation', 'surface_temperature',
    'surface_specific_humidity', 'surface_density',
    'zonal_wind', 'height_500', 'upper_zonal_wind',
    'meridional_wind', 'upper_meridional_wind'
]

# Construct the list of columns to select from df_combined
capacity_factor_columns = [f'{region}' for region in regions]
selected_columns = capacity_factor_columns + climate_variables

# Subset df_combined to only include the selected columns
df_selected = df_combined[selected_columns]

# Calculate correlation matrix only between capacity factors and climate variables
correlation_matrix = df_selected[capacity_factor_columns + climate_variables].corr().loc[capacity_factor_columns, climate_variables]

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Correlation Matrix between Regions (Capacity Factor) and Climate Variables')
plt.xlabel('Climate Variables')
plt.ylabel('Regions (Capacity Factor)')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.show()


# ### Analyzing the relationship between the climate variables and the capacity factor
#
# The code below:
# - does a scatter plot the demand as a function of each climate variable on separate figures,
# - computes the correlation between the capacity factor and each climate variable,
# - computes the correlation matrix between climate variables removing values smaller than 0.3 in absolute value.
#
# > ***Question***
# > - Does their seem to be redundancies between climate variables?
# > - Which climate variables seem to be most relevant to predict the capacity factor?
# > - Discuss the limits of this analysis using correlations alone.

# ## RNN

#

