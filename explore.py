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
 
def inertia_cluster(df):
    ks = range(1,10)
    sse = []
    for k in ks:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)
    print(pd.DataFrame(dict(k=ks, sse=sse)))
    plt.plot(ks, sse, 'bx-')
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.title('optimal k')
    plt.show()