import seaborn as sns
import matplotlib.pyplot as plt
import env
import MySQLdb


def get_db_url(comm = '!', database = '!'):
    db=MySQLdb.connect(host=env.host, user = env.user, \
    passwd = env.password, db=database)
    return psql.read_sql(comm, con=db)


data = db.get_db_url(comm = """Select *
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
AND properties_2017.propertylandusetypeid IN (260, 262,273, 261, 279)
;""", database = 'zillow')

