import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from an Excel file
# Replace 'path_to_your_file.xlsx' with the actual file path
data = pd.read_csv('ZooScan77_metadata.csv')  # Ensure you have the correct file path here

# Ensure the no_images column is numeric
data['no_images'] = pd.to_numeric(data['no_images'], errors='coerce')



# Sort data for better visualization
data_sorted = data.sort_values('no_images', ascending=False)[1:51]


# Set the figure size for better readability
plt.figure(figsize=(14, 60))

# Create a barplot
ax = sns.barplot(x='no_images', y='category_name', data=data_sorted, width=0.8)
ax.set_xscale('log')


ax.tick_params(axis='y', labelsize=9) 
# Add titles and labels
#plt.title('Distribution of Images by Category')
plt.xlabel('Number of Images')
plt.ylabel('Category')

# Show the plot
plt.show()
