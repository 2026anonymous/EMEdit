if 'model_name' not in locals(): model_name = 'metro_at'

config_dict = update_config(config_dict,
    
    # Eval settings (clip)
    # ts2txt
    y_col = 'holiday',
    y_levels = ['This week has no holidays.', 'This week includes a holiday.'],
    y_pred_levels = ['This week has no holidays.', 'This week includes a holiday.'],
    # txt2ts
    txt2ts_y_cols = ['holiday', 'weather', 'temp',],
    # open vocabulary
    open_vocab_dict_path = "../../data/air_quality/aug_text.json",
    
    # Data settings
    seq_length = 168,
    custom_target_cols = ['holiday', 'weather',  'temp', 'label'],
    ts_global_normalize = True, 

    # Model settings
    model_name = model_name,
    
    # Train settings
)
