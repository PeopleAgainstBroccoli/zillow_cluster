import acquire
import prep
import seaborn as sns
import matplotlib.pyplot as plt

data = acquire.wrangle_zillow()
data, subset = prep.prep_zillow()



sns.scatterplot(y="taxvaluedollarcnt", x="logerror", hue = 'fips',
             data=subset)

plt.show()

subset['is_not'] = subset['fips'] != 0
subset = subset.loc[subset.is_not, :]

sns.scatterplot(y="taxvaluedollarcnt", x="logerror", hue = 'fips',
             data=subset)

plt.show()

sns.scatterplot('latitude', 'longitude' , hue = 'taxvaluedollarcnt', data = subset)
plt.show()