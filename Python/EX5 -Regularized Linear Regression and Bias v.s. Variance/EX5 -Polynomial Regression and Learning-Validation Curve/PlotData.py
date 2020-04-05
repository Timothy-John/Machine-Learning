import matplotlib.pyplot as plt

#.....Plotting Data.....
def PlotData(x,y):
    plt.scatter(x,y)
    plt.xlabel('Change in Water Level')
    plt.ylabel('Amount of Water flowing out of Dam')
    plt.title('Data Plot')