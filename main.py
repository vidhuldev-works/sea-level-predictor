from sea_level_predictor import *

def main():

    # scatter plot of the data
    scatter_plot(df)

    # predicted full data plot
    scatter_plot_with_linear_regression(df)

    # predicted filtered data
    filter_data_plot(df)

    # combination 3 plots
    combined_plot(df)

if __name__ == "__main__":
    main()
