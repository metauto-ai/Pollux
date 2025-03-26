## Text Encoder

Support multiple text encoders for prompt embedding

### Environment Requirement

```
pip install ftfy
pip install transformers
```

### Usage

```py
from text_encoder import TextEncoder

# input prompt and model_name ('gemma-2-2b-it-sana' for SANA and 'umt5-xxl' for WAN)
# output: prompt_embeds (shape: [1, 512, 256000] for gemma and [1, 512, 4096] for umt5)

model_name = 'gemma-2-2b-it-sana' 
prompt = 'Write a hello world program'
text_encoder = TextEncoder(model_name)
prompt_embeds = text_encoder.get_t5_prompt_embeds(prompt)
```


