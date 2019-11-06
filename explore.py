import acquire
import seaborn as sns
import matplotlib.pyplot as plt

data = acquire.wrangle_zillow()



subset = data[['latitude', 'longitude', 'taxvaluedollarcnt', 'logerror', 'fips']]
encoder = LabelEncoder()

subset = subset.loc[subset.poor_people, :]
subset = subset.loc[subset.log_me, :]
sns.scatterplot('latitude', 'longitude' , hue = 'taxvaluedollarcnt', data = subset)
plt.show()
sns.violinplot(x = 'logerror', y = 'fips', data = subset)
sns.pairplot(data)



sns.scatterplot(y="taxvaluedollarcnt", x="logerror", hue = 'fips',
             data=subset)



g = sns.FacetGrid(subset, col="logerror",  row="fips")
g = g.map(plt.scatter, "taxvaluedollarcnt")

plt.show()
