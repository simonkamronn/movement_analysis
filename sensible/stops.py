from math import radians, cos, sin, asin, sqrt
import time

from sklearn.cluster import DBSCAN
import pandas as pd


def groupwhile(df, fwhile):
        groups = []
        i = 0
        while i < len(df):
                j = i
                while j < len(df) - 1 and fwhile(i, j + 1):
                        j = j + 1
                groups.append(df.iloc[i:j + 1])
                i = j + 1
        return groups


def haversine(lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        m = 6367000 * c
        return m


def haversine_metric(a, b):
        if len(a) == 2 and len(b) == 2:
                return haversine(lon1=a[0], lat1=a[1], lon2=b[0], lat2=b[1])
        return 0


def getstops_dbscan(user, grp, group_dist=60, dbscan_dist=60, min_deltat=1, min_samples=1):

        groups = groupwhile(grp, lambda start, next: haversine(grp['lon'].values[start],
                                                               grp['lat'].values[start], 
                                                               grp['lon'].values[next],
                                                               grp['lat'].values[next]) <= group_dist)
        values = []
        for g in groups:
                values.append([user, g.lon.median(), g.lat.median(), g.timestamp.values[0], g.timestamp.values[-1]])
        stops = pd.DataFrame(values, columns=['user', 'lon', 'lat', 'arrival', 'departure'])
        stops = stops[stops.departure - stops.arrival >= min_deltat]

        if len(stops) > 0:
                X = stops[['lon', 'lat']].values
                db = DBSCAN(dbscan_dist, min_samples=min_samples, metric=haversine_metric).fit(X)
                stops['label'] = db.labels_
                return stops, db
        else:
                return None
