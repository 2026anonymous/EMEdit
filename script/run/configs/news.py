if 'model_name' not in locals(): model_name = 'news_at'

config_dict = update_config(config_dict,
    
    # Eval settings (clip)
    # ts2txt
    y_col = 'political_news',
    y_levels = ['There are significant political news releases.', 'No significant political news releases.'],
    y_pred_levels = ['No significant political news releases.', 'There are significant political news releases.'],
    # txt2ts
    txt2ts_y_cols = ['political_news', 'SentimentTitle', 'SentimentHeadline', ],
    # open vocabulary
    open_vocab_dict_path = "../../data/news/aug_text.json",
    
    # Data settings
    seq_length = 144,
    custom_target_cols = ['political_news', 'SentimentTitle', 'SentimentHeadline',  'label'],
    ts_global_normalize = True, 

    # Model settings
    model_name = model_name,
    
    # Train settings
)
