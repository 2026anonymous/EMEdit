
# read 
df_train = pd.read_csv('../data/nicu/train.csv.zip', compression='zip')
df_test =  pd.read_csv('../data/nicu/test.csv.zip', compression='zip')
df_left =  pd.read_csv('../data/nicu/left.csv.zip', compression='zip')

if config_dict['open_vocab']:
    df_train, df_test, df_left = gen_open_vocab_text(df_train, df_test, df_left, config_dict)

print('\n\nfinal distribution of text prediction')
print(df_train[config_dict['y_col']].value_counts())
print(df_test[config_dict['y_col']].value_counts())
print(df_left[config_dict['y_col']].value_counts())
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
args0 = {'description_histogram': None,
        'description_ts_event_binary': None
        }

args1 = {'description_histogram': [('description_ts_event_binary', "No events.")],
        'description_ts_event_binary': [('description_histogram', "Low variability.")]
        }

args_ls = [args0, args1]

# Define the base augmentation pairs
base_aug_dict = {'description_histogram': [('Low variability.', 'High variability.'),
                                            ('High variability.', 'Low variability.')],
                'description_ts_event_binary': [('No events.', 'Bradycardia events happened.'),
                                                ('Bradycardia events happened.', 'No events.')],
                }


