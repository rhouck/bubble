import datetime

import pandas as pd
from pandas_datareader import data, wb


start = datetime.datetime(1970, 1, 1)
end = datetime.datetime.now().date()

"""
 Yahoo Finace

 Series: 
 ^GSPC      S&P
 ^DJI       Dow
 ^IXIC      Nasdaq
"""
df = data.DataReader(["^GSPC", '^DJI', '^IXIC'], 'yahoo', start, end)
# confirm series order has not changed
df.columns = ['S&P', 'Dow', 'Nasdaq'] 
df.ix[3,:,:].to_csv('data/market_{0}.csv'.format(str(end).replace("-", "_")))


"""
FRED - St Lois Fed 

Series:
GDP             Gross Domestic Product
CPIAUCSL        Consumer Price Index for All Urban Consumers: All Items
VIXCLS          CBOE Volatility Index: VIX
VXVCLS          CBOE S&P 500 3-Month Volatility Index
AAA             Moody's Seasoned Aaa Corporate Bond Yield
DTB3            3-Month Treasury Bill: Secondary Market Rate
TOTDTEUSQ163N   Total Debt to Equity for United States
GFDEGDQ188S     Federal Debt: Total Public Debt as Percent of Gross Domestic Product
EXHOSLUSM495S   Existing Home Sales
LES1252881600Q  Employed full time: Median usual weekly real earnings: Wage and salary workers: 16 years and over
"""
fred = data.DataReader(["GDP", 
                        "CPIAUCSL", 
                        "CPILFESL", 
                        "VIXCLS", 
                        "VXVCLS", 
                        "AAA", 
                        "DTB3", 
                        "TOTDTEUSQ163N",
                        "GFDEGDQ188S",
                        "EXHOSLUSM495S",
                        "LES1252881600Q"], 
                       "fred", start, end)
fred.to_csv('data/fred.csv'.format(str(end).replace("-", "_")))

"""
 Fama French

 Series: 
"""
# import pandas.io.data as web
# ip = web.DataReader("12_Industry_Portfolios", "famafrench")
# ip.to_csv('data/famma_french.csv'.format(str(end).replace("-", "_")))