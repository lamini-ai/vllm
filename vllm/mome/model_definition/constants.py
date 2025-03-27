WEIGHTS_NAME = "adapter_model.bin"
SAFETENSORS_WEIGHTS_NAME = "adapter_model.safetensors"
CONFIG_NAME = "adapter_config.json"
EMBEDDING_LAYER_NAMES = ["embed_tokens", "lm_head"]
INCLUDE_LINEAR_LAYERS_SHORTHAND = "all-linear"
TOKENIZER_CONFIG_NAME = "tokenizer_config.json"
MOME_ADAPTER_PREFIXES = [
    "mome_attention.attn",
    "mome_attention.query_projection_lora_in",
    "mome_attention.query_projection_lora_out",
    "mome_layer_norm",
    "mlp_lora_in",
    "mlp_lora_out",
    "mlp_layer_norm",
    "lm_head_lora_in",
    "lm_head_lora_out",
]
MOME_KEY_VALUE_PREFIXES = [
    "key_embedding",
    "value_embedding",
]
# The dimentions of sentence-transformers' output
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
SENTENCE_TRANSFORMER_DIM = 384
