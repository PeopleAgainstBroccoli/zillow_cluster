import dbtools as db
import seaborn as sns
import matplotlib.pyplot as plt



def wrangle_zillow():
    return db.get_db_url(comm = """Select *
From properties_2017
Join
(SELECT
p_17.parcelid,
logerror,
transactiondate
FROM predictions_2017 p_17
JOIN 
(SELECT
  parcelid, Max(transactiondate) as tdate
FROM
  predictions_2017
Group By parcelid )as sq1
ON (sq1.parcelid=p_17.parcelid and sq1.tdate = p_17.transactiondate )) sq2
USING (parcelid)
WHERE (latitude IS NOT NULL AND longitude IS NOT NULL)
AND properties_2017.propertylandusetypeid NOT IN (31, 47,246, 247, 248, 267, 290, 291)
LIMIT 300000;""", database = 'zillow')


