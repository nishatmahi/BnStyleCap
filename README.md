# BnStyleCap: Learning Bangla Stylized Image Captioning from Synthetic Supervision

Code and data for BnStyleCap, a two-stage model that generates humorous and romantic Bangla image captions.

## Datasets

| File | Style | Captions |
|---|---|---|
| `data/BnHumor17K.tsv` | Humorous | 17,362 |
| `data/BnRomance17K.tsv` | Romantic | 17,344 |

Each file has three columns:
- `image_id`: the image the factual captions came from.
- `source`: `flickr30k` or `bangladeshi`.
- `caption`: the Bangla style caption.

The captions were written by Gemini from the factual Bangla captions of each image and then checked automatically. They are the training captions used in the paper.

**Images are not included.**
- Flickr30k images can be obtained from the Flickr30k authors under their terms. Their factual Bangla captions come from BanglaView.
- The Bangladeshi photographs are not distributed.

`data/style_guide_bn.md` is the style guide given to the caption generator.

## Model

1. **Stage 1:** a frozen NLLB-CLIP large image encoder, a prefix mapper and a BanglaGPT decoder learn factual captioning.
2. **Stage 2:** a copy of this model is trained for each style on noisy text embeddings of the style captions, mixed with factual image and caption pairs.
3. **Inference:** a modality offset shifts the image embedding before decoding with beam search.

Final settings:
- Stage 2 noise 0.1
- Offset scale 0.5 (humorous) and 0.4 (romantic)
- Beam size 3

## Code

| Script | Purpose |
|---|---|
| `code/extract_features.py` | Encode images and captions with NLLB-CLIP |
| `code/stage1.py` | Train the factual captioner |
| `code/stage2.py` | Train one style model (`--style humorous` or `--style romantic`) |
| `code/generate.py` | Generate captions for test images |
| `code/evaluate.py` | CLIP-S and style accuracy |
| `code/style_classifier.py` | Independent style classifier |

Run each script with `--help` to see its arguments. Default paths point to Kaggle inputs and can be changed on the command line.
## License

The code is released under the MIT License (`LICENSE`). The BnHumor17K and BnRomance17K caption files are released under CC BY 4.0 (`data/LICENSE`). Please cite the BnStyleCap paper when using them.

## Requirements

```
pip install -r requirements.txt
```
