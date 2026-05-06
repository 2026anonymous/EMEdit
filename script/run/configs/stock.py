if 'model_name' not in locals(): model_name = 'stock_at'

config_dict = update_config(config_dict,
    
    # Eval settings (clip)
    # ts2txt
    y_col = 'Emergency',
    y_levels = ['Quiet market environment', 'Sudden event shock'],
    y_pred_levels = ['Sudden event shock', 'Quiet market environment'],
    # txt2ts
    txt2ts_y_cols = ['Emergency', 'Macroeconomics'],
    # open vocabulary
    open_vocab_dict_path = "../../data/stock/aug_text.json",
    
    # Data settings
    seq_length = 60,
    custom_target_cols = ['Emergency', 'Macroeconomics', 'label'],
    ts_global_normalize = True, 

    # Model settings
    model_name = model_name,
    
    # Train settings
)
