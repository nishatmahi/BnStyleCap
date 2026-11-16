# bn_text.py
# Bangla text tokens for NLLB-CLIP. Newer open_clip/transformers versions can no longer set the NLLB source
# language ("Cannot set language for the tokenizer"), so the Bangla tag ben_Beng is silently missing and
# text embeddings stop matching images (CLIP-S about 0). This builds the NLLB input by hand:
#   [ben_Beng] tokens ... [</s>] [pad ...]   (NLLB source format), always, whatever the library version.
import torch

def bn_tokenize(t, texts, lang='ben_Beng'):
    hf = t.tokenizer
    L = getattr(t, 'context_length', None) or 64
    lang_id = hf.convert_tokens_to_ids(lang)
    assert lang_id is not None and lang_id != hf.unk_token_id, f'{lang} not in tokenizer vocab'
    body = hf(list(texts), add_special_tokens=False, truncation=True, max_length=L - 2)['input_ids']
    out = torch.full((len(texts), L), hf.pad_token_id, dtype=torch.long)
    for k, ids in enumerate(body):
        seq = [lang_id] + ids + [hf.eos_token_id]
        out[k, :len(seq)] = torch.tensor(seq)
    return out
