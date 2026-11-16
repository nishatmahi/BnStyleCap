"""Paired significance, ours vs Gemma 3 12B (same images): per-caption BERTScore (BanglaBERT layer 9, Gemini
references on 1000 images and human references on 212) and CLIP-S (NLLB-CLIP large, 2.5*max(cos,0), 1000 images).
Wilcoxon signed-rank p-value, as in significance.py.

  python tools/significance_gemma.py --gemma_dir <dir with gemma_{style}_1000.json> --img_dir <Images>
"""
import argparse, csv, json, os, sys, unicodedata
sys.modules['sklearn'] = None
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
import numpy as np
from scipy.stats import wilcoxon

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
EV = os.path.join(ROOT, 'eval_upload')
OFF = {'humorous': '0.5', 'romantic': '0.4'}
nfc = lambda s: ' '.join(unicodedata.normalize('NFC', s).split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gemma_dir', default=os.path.expanduser('~/Downloads'))
    ap.add_argument('--img_dir', default=os.path.join(ROOT, 'data', 'Images'))
    ap.add_argument('--clip_only', action='store_true')
    a = ap.parse_args()
    rows = {r['image_id']: r for r in csv.DictReader(open(f'{EV}/test_captions_1000.csv', encoding='utf-8-sig'))}
    hum = {r['image_id']: r for r in csv.DictReader(open(f'{EV}/human_refs_212_final.csv', encoding='utf-8-sig'))}
    ids, hids = list(rows), [i for i in hum if i in rows]
    sysc = {}
    for st in OFF:
        sysc[st] = {'ours': {x['image_id']: nfc(x[st]) for x in json.load(open(f'{EV}/gen_{st}_c18k_beam3_off{OFF[st]}.json', encoding='utf-8'))},
                    'gemma': {x['image_id']: nfc(x[st]) for x in json.load(open(os.path.join(a.gemma_dir, f'gemma_{st}_1000.json'), encoding='utf-8'))}}
    out = ['| Metric | Style | Ours | Gemma | Wilcoxon p | n |', '|---|---|---|---|---|---|']

    def add(name, st, x, y):
        out.append(f'| {name} | {st} | {np.mean(x):.4f} | {np.mean(y):.4f} | {wilcoxon(x, y).pvalue:.1e} | {len(x)} |')
        print(out[-1], flush=True)

    if a.clip_only: out = [l for l in open(os.path.join(ROOT, 'significance_gemma.md'), encoding='utf-8').read().splitlines() if l] if os.path.exists(os.path.join(ROOT, 'significance_gemma.md')) else out
    import transformers
    _orig = transformers.AutoTokenizer.from_pretrained
    def _tok(*k, **kw):
        t = _orig(*k, **kw); t.model_max_length = 512; return t
    transformers.AutoTokenizer.from_pretrained = _tok
    from bert_score import score
    bs = lambda c, r: 100 * score(c, r, model_type='csebuetnlp/banglabert', num_layers=9, batch_size=32, lang='bn')[2].numpy()
    for st, hc in (() if a.clip_only else (('humorous', 'bangla_humorous'), ('romantic', 'bangla_romantic'))):
        ref = [nfc(rows[i][st + '_reference']) for i in ids]
        add('BERTScore LLM refs', st, bs([sysc[st]['ours'][i] for i in ids], ref), bs([sysc[st]['gemma'][i] for i in ids], ref))
        ref = [nfc(hum[i][hc]) for i in hids]
        add('BERTScore human refs', st, bs([sysc[st]['ours'][i] for i in hids], ref), bs([sysc[st]['gemma'][i] for i in hids], ref))

    import torch
    from PIL import Image
    from open_clip import create_model_from_pretrained, get_tokenizer
    model, prep = create_model_from_pretrained('nllb-clip-large-siglip', 'v1'); model.eval()
    tok = get_tokenizer('nllb-clip-large-siglip')
    cache = os.path.join(ROOT, 'nllbclip_test_img.pt'); img = []
    if os.path.exists(cache): img = [torch.load(cache)]
    with torch.no_grad():
        for k in (range(0) if img else range(0, len(ids), 16)):
            f = model.encode_image(torch.stack([prep(Image.open(os.path.join(a.img_dir, i + '.jpg')).convert('RGB')) for i in ids[k:k + 16]]))
            img.append(f / f.norm(dim=-1, keepdim=True))
            print('images', k + 16, flush=True)
    img = torch.cat(img); torch.save(img, cache)

    def clips(caps):
        out_ = []
        with torch.no_grad():
            for k in range(0, len(caps), 64):
                f = model.encode_text(tok(caps[k:k + 64])); out_.append(f / f.norm(dim=-1, keepdim=True))
        return (2.5 * (torch.cat(out_) * img).sum(-1).clamp(min=0)).numpy()
    for st in OFF:
        add('CLIP-S', st, clips([sysc[st]['ours'][i] for i in ids]), clips([sysc[st]['gemma'][i] for i in ids]))
    open(os.path.join(ROOT, 'significance_gemma.md'), 'w', encoding='utf-8').write('\n'.join(out) + '\n')
    print('saved significance_gemma.md')


if __name__ == '__main__':
    main()
