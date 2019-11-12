#import acquire
import prep
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

#data = acquire.wrangle_zillow()
#data = prep.prep_zillow()



def remove_county(county_num):
    data['is_not'] = data['fips'] != county_num
    return data.loc[data.is_not, :]



def tax_log(data):
    sns.scatterplot(y="taxvaluedollarcnt", x="logerror", hue = 'fips',
             data=data)
    plt.show()


def tax_val(data):
    sns.scatterplot('latitude', 'longitude' , hue = 'taxvaluedollarcnt', data = data)
    plt.show()

def ammen_count(data):
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

def mean_logerror_county(data):
    data['county'] = data['fips'].apply(lambda x : {0 : 'LA', 0.5 : 'Orange_County', 1 : 'Ventura'}.get(x, ' '))
    print('<<<<|_AVERAGE_LOG_ERROR_PER_COUNTY_|>>>>>')
    print(data.groupby('county')['logerror'].mean())