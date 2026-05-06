from modelscope import snapshot_download

# 下载 Qwen3-Embedding-4B（推荐）
model_dir = snapshot_download(
    model_id='',
    cache_dir='./',      # 保存路径
    revision='master'
)

print(model_dir)