import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading model predictions...")
# Read prediction results from the previous phase
df = pd.read_csv('model_predictions.csv')

# Set plot dimensions and style parameters
plt.figure(figsize=(8, 6))
sns.set_style("whitegrid")

# Plot scatter points
sns.scatterplot(x='Experimental_pIC50', y='Predicted_pIC50', data=df, alpha=0.5, color='royalblue', edgecolor='w', s=60)

# Plot the ideal diagonal prediction line
# If the model were 100% accurate, all points would fall on this red line
min_val = min(df['Experimental_pIC50'].min(), df['Predicted_pIC50'].min())
max_val = max(df['Experimental_pIC50'].max(), df['Predicted_pIC50'].max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Ideal Prediction (Y = X)')

# Set layout titles and axis labels
plt.title('QSAR Model Performance: Experimental vs Predicted pIC50\nTarget: Acetylcholinesterase (AChE)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Experimental pIC50 (Actual)', fontsize=12, fontweight='bold')
plt.ylabel('Predicted pIC50 (Model)', fontsize=12, fontweight='bold')
plt.legend()

# Save the high-resolution plot
plt.savefig('qsar_scatter_plot.png', dpi=300, bbox_inches='tight')
print("Plot successfully generated and saved as 'qsar_scatter_plot.png'")

# Display the plot
plt.show()
