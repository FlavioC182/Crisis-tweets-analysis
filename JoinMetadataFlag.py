# This code snippet is used to join csv (dataframes) created using the other codes.
# The join is between MetaData dataframe and Flags + text dataframe

import pandas as pd
import numpy as np
from Utilities import joinBetweenMetaDataAndCities
PakistanMetaData = 'https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/MetaData2/2013_pakistan_metadati.csv'
PakistanIrrelevevantMD = 'https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/MetaData2/2013_pakistan_irrelevant_metadati.csv'
PakistanFlags = 'https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/MetaData2/2013_pakistan_Flags_NLP.csv'
CaliforniaMetaData = 'https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/MetaData2/2014_california_metadati.csv'
CaliforniaFlags = 'https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/MetaData2/2014_california_Flags_NLP.csv'
CaliforniaIrrelevantMD = 'https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/MetaData2/2014_california_irrelevant_metadati.csv'
ChileMetaData = 'https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/MetaData2/2014_chile_metadati.csv'
ChileFlags = 'https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/MetaData2/2014_chile_Flags_NLP.csv'
ChileIrrelevantMD = 'https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/MetaData2/2014_chile_irrelevant_metadati.csv'
MexicoMetaData = 'https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/MetaData2/2014_odile_hurricane_metadati.csv'
MexicoFlags = 'https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/MetaData2/2014_odile_hurricane_Flags_NLP.csv'
MexicoIrrelevantMD = 'https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/MetaData2/2014_odile_hurricane_irrelevant_metadati.csv'
IndiaFloodsMetaData = 'https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/MetaData2/2014_india_floods_metadati.csv'
IndiaFloodsIrrelevantMD = 'https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/MetaData2/2014_india_floods_irrelevant_metadati.csv'
IndiaFloodsFlags = 'https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/MetaData2/2014_india_Flags_NLP.csv'
Pakistan2014MetaData = 'https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/MetaData2/2014_pakistan_floods_metadati.csv'
Pakistan2014IrrelevantMD = 'https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/MetaData2/2014_pakistan_floods_irrelevant_metadati.csv'
Pakistan2014Flags = 'https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/MetaData2/2014_pakistan_flood_Flags_NLP.csv'
HagupitMetaData = 'https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/MetaData2/2014_hagupit_typhoon_metadati.csv'
HagupitIrrelevantMD = 'https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/MetaData2/2014_hagupit_typhoon_irrelevant_metadati.csv'
HagupitFlags = 'https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/MetaData2/2014_hagupit_typhoon_Flags_NLP.csv'
PamMetaData = 'https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/MetaData2/2015_pam_cyclone_metadati.csv'
PamIrrelevantMD = 'https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/MetaData2/2015_pam_cyclone_irrelevant_metadati.csv'
PamFlags = 'https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/MetaData2/2015_pam_cyclone_Flags_NLP.csv'

MetaDataF = pd.read_csv(PamIrrelevantMD, header=0)
FlagsDataF = pd.read_csv(PamFlags, header=0)

JoinDataFrame = joinBetweenMetaDataAndCities(MetaDataF, FlagsDataF)
JoinDataFrame.to_csv(r'Metadata2/2015_pam_cyclone_irrelevant_metadati_Flags.csv',
                     header=JoinDataFrame.columns.values, index=True, sep=',', mode='w')
