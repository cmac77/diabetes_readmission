#%% Exploratory Data Analysis Module

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from pyprojroot import here
import sys
import yaml

# Set the project root directory using here() and add it to sys.path
path_root = here()
if str(path_root) not in sys.path:
    sys.path.insert(0, str(path_root))

# Load configurations using config.py
from src.config import load_config

# Set Seaborn style for plots
sns.set_style("whitegrid")

#%% Load Configuration
path_config = here() / "configs" / "config_settings.yaml"
with open(path_config, 'r') as file:
    dict_config = yaml.safe_load(file)

path_data_processed = Path(here()) / dict_config['paths']['data_cleaned']
path_figures = Path(here()) / dict_config['paths']['results'] / "figures"
path_tables = Path(here()) / dict_config['paths']['results'] / "tables"

# Ensure output directories exist
path_figures.mkdir(parents=True, exist_ok=True)
path_tables.mkdir(parents=True, exist_ok=True)

#%% Function: Generate Summary Statistics
def summary_statistics(df):
    """
    Generates and saves descriptive statistics for numerical and categorical features.
    """
    # Numerical summary
    df_numerical_summary = df.describe()
    df_numerical_summary.to_csv(path_tables / "numerical_summary.csv")
    
    # Categorical summary
    df_categorical_summary = df.describe(include='category')
    df_categorical_summary.to_csv(path_tables / "categorical_summary.csv")
    
    print("Summary statistics saved to results/tables.")
#%% Function: Plot Feature Distributions with Log Scale
#%% Function: Plot Feature Distributions with Dynamic y-axis Limits
def plot_distributions(df):
    """
    Plots and saves the distribution of numerical features and frequency of categorical features
    with dynamically adjusted y-axis limits for better visibility.
    """
    # Plot numerical features
    list_num_features = df.select_dtypes(include=['int64', 'float64']).columns
    fig, axes = plt.subplots(len(list_num_features) // 2 + len(list_num_features) % 2, 2, figsize=(15, 4 * len(list_num_features) // 2))
    axes = axes.flatten()
    
    for i, feature in enumerate(list_num_features):
        sns.histplot(df[feature], kde=True, bins=30, ax=axes[i])
        axes[i].set_title(f'Distribution of {feature}')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(path_figures / "numerical_distributions.png")
    plt.close()

    # Plot categorical features with dynamic y-axis limits
    list_cat_features = df.select_dtypes(include='category').columns
    num_columns = 3  # Adjust the number of columns here
    num_rows = len(list_cat_features) // num_columns + (len(list_cat_features) % num_columns > 0)
    
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(20, 4 * num_rows))
    axes = axes.flatten()
    
    for i, feature in enumerate(list_cat_features):
        ax = axes[i]
        sns.countplot(x=feature, data=df, ax=ax)
        
        # Calculate the max count for the current feature
        max_count = df[feature].value_counts().max()
        
        # Set a reasonable y-axis limit, using log scale if necessary
        if max_count > 1000:  # Threshold to switch to log scale
            ax.set_yscale('log')
            ax.set_ylabel('Count (log scale)')
        else:
            ax.set_ylim(0, max_count * 1.1)  # Add 10% padding for better visibility
            ax.set_ylabel('Count')
        
        ax.set_title(f'Frequency of {feature}')
        ax.set_xlabel(feature)
        ax.tick_params(axis='x', rotation=45)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(path_figures / "categorical_distributions_dynamic.png")
    plt.close()

    print("Distribution plots with dynamic y-axis limits saved to results/figures.")



#%% Function: Correlation Analysis
def correlation_analysis(df):
    """
    Visualizes and saves the correlation between numerical features using a heatmap.
    """
    list_num_features = df.select_dtypes(include=['int64', 'float64']).columns
    matrix_corr = df[list_num_features].corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(matrix_corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title("Correlation Heatmap")
    plt.savefig(path_figures / "correlation_heatmap.png")
    plt.close()

    print("Correlation heatmap saved to results/figures.")

#%% Function: Analyze Missing Data
def missing_data_analysis(df):
    """
    Analyzes and visualizes missing data patterns in the dataset.
    """
    series_missing_values = df.isnull().sum()
    series_missing_values = series_missing_values[series_missing_values > 0].sort_values(ascending=False)
    
    # Save missing values table
    series_missing_values.to_csv(path_tables / "missing_values.csv")
    
    # Visualize missing values
    plt.figure(figsize=(12, 6))
    sns.barplot(x=series_missing_values.index, y=series_missing_values.values)
    plt.title("Missing Values by Feature")
    plt.xlabel("Feature")
    plt.ylabel("Count of Missing Values")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(path_figures / "missing_values.png")
    plt.close()

    print("Missing data analysis saved to results/tables and results/figures.")

#%% Main Function: Run EDA
def run_eda():
    """
    Runs the full EDA pipeline on the cleaned DataFrame loaded from a Parquet file.
    """
    df_cleaned = pd.read_pickle(path_data_processed)
    
    print("Running Summary Statistics...")
    summary_statistics(df_cleaned)
    
    print("\nPlotting Distributions...")
    plot_distributions(df_cleaned)
    
    print("\nAnalyzing Correlations...")
    correlation_analysis(df_cleaned)
    
    print("\nAnalyzing Missing Data...")
    missing_data_analysis(df_cleaned)

#%% Execute EDA
if __name__ == "__main__":
    run_eda()

#%%
