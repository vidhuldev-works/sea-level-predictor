import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('epa-sea-level.csv')
#df.info()
#print(df.head())

#function to create a scatter plot of the data
def scatter_plot(dataframe):

    #handles null value by interpolate
    dataframe['CSIRO Adjusted Sea Level'].interpolate(inplace=True)

    #plot the data
    plt.scatter(dataframe['Year'], dataframe['CSIRO Adjusted Sea Level'], color= 'blue', s= 10)
    plt.grid(True)
    plt.legend()

    #set title and labels
    plt.title('Global Sea Level Rise (1880-2014) Based on CSIRO Adjusted Data')
    plt.xlabel('Year')
    plt.ylabel('CSIRO Adjusted Sea Level')

    #show the plot
    plt.show()

#scatter_plot(df)

from scipy.stats import linregress

def scatter_plot_with_linear_regression(dataframe):
    """
    Creates a scatter plot using the Year column as the x-axis and the CSIRO Adjusted Sea Level column as the y-axis.
    Fits a linear regression line to the data and extends the prediction to the year 2050.
    Handles null values by dropping rows with NaN.
    """
    # Drop rows with NaN or infinite values
    dataframe = dataframe.dropna(subset=['Year', 'CSIRO Adjusted Sea Level'])
    dataframe = dataframe[np.isfinite(dataframe['CSIRO Adjusted Sea Level'])]
    
    # Scatter plot for observed data (1880-2014)
    plt.scatter(dataframe['Year'], dataframe['CSIRO Adjusted Sea Level'], color='blue', s=10, label='Observed Data')

    # Linear regression using linregress
    slope, intercept, r_value, p_value, std_err = linregress(dataframe['Year'], dataframe['CSIRO Adjusted Sea Level'])

    # Print the slope and intercept for debugging
    print(f"Slope: {slope}, Intercept: {intercept}")

    # Create an array of years from 1880 to 2050 for prediction
    years_for_prediction = np.arange(1880, 2051)

    # Predict sea levels for these years using the linear model (y = mx + b)
    predicted_sea_levels = slope * years_for_prediction + intercept

    # Plot the linear regression line
    plt.plot(years_for_prediction, predicted_sea_levels, color='red', label='Linear Regression (2050 Prediction)')

    # Highlight the predicted value in 2050 for the filtered data
    plt.scatter(2050, predicted_sea_levels[-1], color='red', zorder=5)
    plt.text(2050, predicted_sea_levels[-1], f'{predicted_sea_levels[-1]:.2f}', color='red')

    # Set title and labels
    plt.title('Global Sea Level Rise Observed Data with Predicted Data (1880-2050) Based on CSIRO Adjusted Data')
    plt.xlabel('Year')
    plt.ylabel('CSIRO Adjusted Sea Level')

    # Show legend and grid
    plt.legend()
    plt.grid(True)

    # Show the combined plot
    plt.show()

#scatter_plot_with_linear_regression(df)

def filter_data_plot(dataframe):

    # Drop rows with NaN or infinite values
    dataframe = dataframe.dropna(subset=['Year', 'CSIRO Adjusted Sea Level'])
    dataframe = dataframe[np.isfinite(dataframe['CSIRO Adjusted Sea Level'])]

    # filter data since 2000
    filtered_data = dataframe[dataframe['Year'] >= 2000]

    # linear regression using linregress
    slope, intercept, r_value, p_value, std_err = linregress(filtered_data['Year'], filtered_data['CSIRO Adjusted Sea Level'])

    # predicting the data
    years = np.arange(2000, 2051)
    predict_sea_level = slope * years + intercept

    # scatter plot for the observed data
    plt.scatter(dataframe['Year'], dataframe['CSIRO Adjusted Sea Level'], color= 'blue', s= 10, label= 'Observed Data')

    # line plot for predicted data
    plt.plot(years, predict_sea_level, color= 'green', linestyle= '--', label= 'Predicted (2000 - 2050)')

    # Highlight predicted value in 2050
    plt.scatter(2050, predict_sea_level[-1], color='red')  # Highlight predicted value in 2050
    plt.text(2050, predict_sea_level[-1], f'{predict_sea_level[-1]:.2f}', color='red')

    plt.xlabel('Year')
    plt.ylabel('CSIRO Adjusted Sea Level')
    plt.title('Filtered and Predicted Global Sea Level Rise Data (2000-2050) Based on CSIRO Adjusted Data')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # show the combined plot
    plt.show()

#filter_data_plot(df)

def combined_plot(dataframe):

    # Drop rows with NaN or infinite values
    dataframe = dataframe.dropna(subset=['Year', 'CSIRO Adjusted Sea Level'])
    dataframe = dataframe[np.isfinite(dataframe['CSIRO Adjusted Sea Level'])]

    # scatter plot for the observed data
    plt.scatter(dataframe['Year'], dataframe['CSIRO Adjusted Sea Level'], color= 'blue', s= 10, label= 'Observed Data')

    # !.linear regression on full data
    slope_all, intercept_all, r_value, p_value, std_err = linregress(dataframe['Year'], dataframe['CSIRO Adjusted Sea Level'])

    # predict full linear regression line
    years_full = np.arange(1880, 2051)
    sea_level_full = slope_all * years_full + intercept_all

    # plot the full linear regression
    plt.plot(years_full, sea_level_full, color= 'red', label= 'Predicted Full Data (1880 - 2050)', linewidth= 2)

    # 2.linear regression on data from 2000
    slope_filtered, intercept_filtered, r_value_filtered, p_value_filtered, std_err_filtered = linregress(dataframe['Year'], dataframe['CSIRO Adjusted Sea Level'])

    # predict filtered linear regression line
    years_filtered = np.arange(2000, 2051)
    sea_level_filtered = slope_filtered * years_filtered + intercept_filtered

    # plot the filteres linear regression
    plt.plot(years_filtered, sea_level_filtered, color= 'yellow', linestyle= '--', label= 'predicted Filtered Data (2000 - 2050)', linewidth= 1.5)

    # Highlight the predicted value in 2050 for the filtered data
    plt.scatter(2050, sea_level_filtered[-1], color='red', zorder=5)
    plt.text(2050, sea_level_filtered[-1], f'{sea_level_filtered[-1]:.2f}', color='red')

    # plot info
    plt.xlabel('Year')
    plt.ylabel('Sea Level (inches)')
    plt.title('Rise in Sea Level')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # show the plot
    plt.show()

#combined_plot(df)