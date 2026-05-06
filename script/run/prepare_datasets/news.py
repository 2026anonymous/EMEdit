# ------------------------------------------------------------------------------------------------
# prepare dataset and arguments for training
# ------------------------------------------------------------------------------------------------
# from sklearn.model_selection import train_test_split

# df = pd.read_csv('../../data_raw/air_quality/air_quality.csv.zip', compression='zip')
# df.columns = df.columns.astype(str)
# df['text'] = ''
# for str_col in config_dict['txt2ts_y_cols']:
#     df['text'] += ' ' + df[str_col]
# df['text'] = df['text'].str.strip()
# df_train, df_temp = train_test_split(df, test_size=0.3, stratify=df[config_dict['y_col']], random_state=config_dict['random_state'])
# df_test, df_left = train_test_split(df_temp, test_size=1/3,stratify=df_temp[config_dict['y_col']],random_state=config_dict['random_state'])
        
# if config_dict['downsample']:
#     df_train = downsample_neg_levels(df_train, config_dict, config_dict['random_state'])
#     df_test = downsample_neg_levels(df_test, config_dict, config_dict['random_state'])
#     df_left = downsample_neg_levels(df_left, config_dict, config_dict['random_state'])
    

# df_train['label'] = df_train.index.to_series()
# df_test['label'] = df_test.index.to_series()
# df_left['label'] = df_left.index.to_series()

# # Save the dataframes as compressed CSV files
# df_train.to_csv('../../data/air_quality/train.csv.zip', compression='zip', index=False)
# df_test.to_csv('../../data/air_quality/test.csv.zip', compression='zip', index=False)
# df_left.to_csv('../../data/air_quality/left.csv.zip', compression='zip', index=False)

# read 
df_train = pd.read_csv('../data/news/train.csv.zip', compression='zip')
df_test =  pd.read_csv('../data/news/test.csv.zip', compression='zip')
df_left =  pd.read_csv('../data/news/left.csv.zip', compression='zip')

if config_dict['open_vocab']:
    df_train, df_test, df_left = gen_open_vocab_text(df_train, df_test, df_left, config_dict)

print('\n\nfinal distribution of text prediction')
print(df_train['text'].value_counts())
print(df_test['text'].value_counts())
print(df_left['text'].value_counts())




# ------------------------------------------------------------------------------------------------
# prepare arguments for evaluation
# ------------------------------------------------------------------------------------------------
df_eval = df_left
w = 0.8 # stength of augmentation

math = True
ts_dist = True
rats = True

# argument dictionary {y_col:conditions}
args0 = {'Topic': None,
         'SentimentTitle': None,
         'SentimentHeadline': None,
         'political_news':None
        }

args1 = {'Topic': [('SentimentTitle', 'The title of news has neutral sentiment.'),('SentimentHeadline', 'The headline of news has negative sentiment.'),('political_news', 'There are significant political news releases.')],
        'SentimentTitle': [('Topic', 'The topic of news is economy.'),('SentimentHeadline', 'The headline of news has positive sentiment.'),('political_news', 'No significant political news releases.')],
        'SentimentHeadline': [('Topic', 'The topic of news is microsoft.'),('SentimentTitle', 'The title of news has neutral sentiment.'),('political_news', 'No significant political news releases.')],
        'political_news': [('Topic', 'The topic of news is economy.'),('SentimentTitle', 'The title of news has neutral sentiment.'),('SentimentHeadline', 'The headline of news has positive sentiment.')]
        }

args_ls = [args0, args1]

# Define the base augmentation pairs used in math and ts_dist
base_aug_dict = {'Topic': [('The topic of news is economy.', 'The topic of news is microsoft.'),
                             ('The topic of news is microsoft.', 'The topic of news is economy.')],
                'SentimentTitle': [('The title of news has neutral sentiment.', 'The title of news has positive sentiment.'),
                            ('The title of news has neutral sentiment.', 'The title of news has negative sentiment.'),
                            ('The title of news has positive sentiment.', 'The title of news has neutral sentiment.'),
                            ('The title of news has positive sentiment.', 'The title of news has negative sentiment.'),
                            ('The title of news has negative sentiment.', 'The title of news has positive sentiment.'),
                            ('The title of news has negative sentiment.', 'The title of news has neutral sentiment.')],
                'SentimentHeadline': [('The headline of news has neutral sentiment.',  'The headline of news has positive sentiment.'),
                                      ('The headline of news has neutral sentiment.',  'The headline of news has negative sentiment.'),
                                      ('The headline of news has positive sentiment.', 'The headline of news has neutral sentiment.'),
                                      ('The headline of news has positive sentiment.', 'The headline of news has negative sentiment.'),
                                      ('The headline of news has negative sentiment.', 'The headline of news has positive sentiment.'),
                                      ('The headline of news has negative sentiment.', 'The headline of news has neutral sentiment.')],
                'political_news': [ ('There are significant political news releases.', 'No significant political news releases.'),
                                    ('No significant political news releases.', 'There are significant political news releases.')],
                }


