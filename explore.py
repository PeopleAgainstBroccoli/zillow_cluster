import acquire
import prep
import seaborn as sns
import matplotlib.pyplot as plt

data = acquire.wrangle_zillow()
data = prep.prep_zillow()



sns.scatterplot(y="taxvaluedollarcnt", x="logerror", hue = 'fips',
             data=subset)

plt.show()





data['is_not'] = data['fips'] != 0
data = data.loc[data.is_not, :]

sns.scatterplot(y="taxvaluedollarcnt", x="logerror", hue = 'fips',
             data=data)

plt.show()

sns.scatterplot('latitude', 'longitude' , hue = 'taxvaluedollarcnt', data = data)
plt.show()



sns.lineplot(x = 'ammenity_count', y = 'logerror', data = data)
plt.show()
 
