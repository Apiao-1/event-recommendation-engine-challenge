import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 200)

if __name__ == '__main__':
    DATA_PATH = "data/"
    # attendees = pd.read_csv(DATA_PATH + 'event_attendees.csv')
    event = pd.read_csv(DATA_PATH + 'events.csv')[:10000]
    event.to_csv(DATA_PATH + 'events_small.csv', index=False)

    # train = pd.read_csv(DATA_PATH + 'train.csv')
    # friends = pd.read_csv(DATA_PATH + 'user_friends.csv')
    # user = pd.read_csv(DATA_PATH + 'users.csv')

    # print(attendees.shape) # (24144, 5)
    # print(attendees.head())
    # print(event.shape) # (3137972, 110)
    # print(event.head())
    # print(train.shape) # (15398, 6)
    # print(train.head())
    # print(friends.shape) # (38202, 2)
    # print(friends.head())
    # print(user.shape) # (38209, 7)
    # print(user.head())

    loc_dict2 = {}
    chunks = event
    latlngdict = {}
    count = 0
    for chunk in chunks.iterrows():
        for e in chunk.iterrows():
            e = e[1]
            eid = int(e['event_id'])
            city = e['city']
            state = e['state']
            country = e['country']
            lat = e['lat']
            lng = e['lng']
            if isinstance(city, float):
                city = None
            if isinstance(state, float):
                state = None
            if isinstance(country, float):
                country = None
            # if isnan(lat) or isnan(lng):
            #     lat = None
            #     lng = None
            if not (city or state or country or lat or lng):
                continue
            d = {}
            if city:
                d['city'] = city
            if state:
                d['state'] = state
            if country:
                if country == 'Democratic Republic Congo':
                    country = 'Democratic Republic of the Congo'
                d['country'] = country
            d['lat'] = lat
            d['lng'] = lng
            if lat:
                if city:
                    latlngdict[(lat, lng)] = (city, country, state)
                else:
                    count += 1
            loc_dict2[eid] = d
            print(loc_dict2)

    '''
    (24144, 5)
        event                                                yes                                              maybe                                            invited                     no
0  1159822043  1975964455 252302513 4226086795 3805886383 142...  2733420590 517546982 1350834692 532087573 5831...  1723091036 3795873583 4109144917 3560622906 31...  3575574655 1077296663
1   686467261  2394228942 2686116898 1056558062 3792942231 41...  1498184352 645689144 3770076778 331335845 4239...  1788073374 733302094 1830571649 676508092 7081...                    NaN
2  1186208412                                                NaN                              3320380166 3810793697                               1379121209 440668682  1728988561 2950720854
3  2621578336                                                NaN                                                NaN                                                NaN                    NaN
4   855842686  2406118796 3550897984 294255260 1125817077 109...  2671721559 1761448345 2356975806 2666669465 10...  1518670705 880919237 2326414227 2673818347 332...             3500235232
(3137972, 110)
     event_id     user_id                start_time city state  zip country  lat  lng  c_1  c_2  c_3  c_4  c_5  c_6  c_7  c_8  c_9  c_10  c_11  c_12  c_13  c_14  c_15  c_16  c_17  c_18  c_19  c_20  c_21  c_22  c_23  c_24  c_25  c_26  c_27  c_28  c_29  c_30  c_31  c_32  c_33  c_34  c_35  c_36  c_37  c_38  c_39  c_40  c_41  c_42  c_43  c_44  c_45  c_46  c_47  c_48  c_49  c_50  c_51  c_52  c_53  c_54  c_55  c_56  c_57  c_58  c_59  c_60  c_61  c_62  c_63  c_64  c_65  c_66  c_67  c_68  c_69  c_70  c_71  c_72  c_73  c_74  c_75  c_76  c_77  c_78  c_79  c_80  c_81  c_82  c_83  c_84  c_85  c_86  c_87  c_88  c_89  c_90  c_91  c_92  c_93  c_94  c_95  c_96  c_97  c_98  c_99  c_100  c_other
0   684921758  3647864012  2012-10-31T00:00:00.001Z  NaN   NaN  NaN     NaN  NaN  NaN    2    0    2    0    0    0    0    0    0     0     0     0     0     0     1     0     0     0     0     0     0     0     0     0     0     0     0     0     1     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     1     0     0     0     0     0     0      0        9
1   244999119  3476440521  2012-11-03T00:00:00.001Z  NaN   NaN  NaN     NaN  NaN  NaN    2    0    2    0    0    0    0    0    0     0     0     0     0     0     1     0     0     0     0     0     0     0     0     0     0     0     0     0     1     0     0     0     0     0     2     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     1     0     0     0     1     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0      0        7
2  3928440935   517514445  2012-11-05T00:00:00.001Z  NaN   NaN  NaN     NaN  NaN  NaN    0    0    0    0    0    0    0    0    0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     2     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0      0       12
3  2582345152   781585781  2012-10-30T00:00:00.001Z  NaN   NaN  NaN     NaN  NaN  NaN    1    0    2    1    0    0    0    0    0     0     0     0     0     0     2     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     1     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     1     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0      0        8
4  1051165850  1016098580  2012-09-27T00:00:00.001Z  NaN   NaN  NaN     NaN  NaN  NaN    1    1    0    0    0    0    0    2    0     0     0     0     0     0     1     0     0     0     1     2     0     0     0     0     0     0     0     0     2     0     0     1     0     0     1     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     1     1     0     0     0     0     0     0     0     1     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0      0        9
(15398, 6)
      user       event  invited                         timestamp  interested  not_interested
0  3044012  1918771225        0  2012-10-02 15:53:05.754000+00:00           0               0
1  3044012  1502284248        0  2012-10-02 15:53:05.754000+00:00           0               0
2  3044012  2529072432        0  2012-10-02 15:53:05.754000+00:00           1               0
3  3044012  3072478280        0  2012-10-02 15:53:05.754000+00:00           0               0
4  3044012  1390707377        0  2012-10-02 15:53:05.754000+00:00           0               0
(38202, 2)
         user                                            friends
0  3197468391  1346449342 3873244116 4226080662 1222907620 54...
1  3537982273  1491560444 395798035 2036380346 899375619 3534...
2   823183725  1484954627 1950387873 1652977611 4185960823 42...
3  1872223848  83361640 723814682 557944478 1724049724 253059...
4  3429017717  4253303705 2130310957 1838389374 3928735761 71...
(38209, 7)
      user_id locale birthyear  gender                  joinedAt            location  timezone
0  3197468391  id_ID      1993    male  2012-10-02T06:40:55.524Z    Medan  Indonesia     480.0
1  3537982273  id_ID      1992    male  2012-09-29T18:03:12.111Z    Medan  Indonesia     420.0
2   823183725  en_US      1975    male  2012-10-06T03:14:07.149Z  Stratford  Ontario    -240.0
3  1872223848  en_US      1991  female  2012-11-04T08:59:43.783Z        Tehran  Iran     210.0
4  3429017717  id_ID      1995  female  2012-09-10T16:06:53.132Z                 NaN     420.0
'''
