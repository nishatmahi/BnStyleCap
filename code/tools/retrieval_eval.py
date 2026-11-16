"""Caption-to-image retrieval on the 1,000 test images (image specificity of each caption set).

For every caption, all test images are ranked by cosine similarity in the NLLB-CLIP large space (the same
encoder and image features as CLIP-S in the paper). R@1 / R@5 = share of captions whose own image is ranked
first / within the top five. CLIP-S (2.5 * max(cos, 0) with the own image) is reported as well.

Caption sets: human factual reference (BanglaView, first caption), Gemini humorous and romantic references,
Qwen2.5-VL, Claude and our model (humorous 0.5, romantic 0.4, beam 3).

Kaggle:  python tools/retrieval_eval.py --test_img <.../style_feats_large/test_images.pt> --out <dir>
Local check without models:  python tools/retrieval_eval.py --dummy
"""
import argparse, csv, glob, json, os, sys, unicodedata
import numpy as np

SETS = [  # name, file, field
    ('Human factual reference', 'test_captions_1000.csv', 'factual_reference'),
    ('Gemini humorous reference', 'test_captions_1000.csv', 'humorous_reference'),
    ('Gemini romantic reference', 'test_captions_1000.csv', 'romantic_reference'),
    ('Qwen2.5-VL humorous', 'qwen_humorous_150.json', 'humorous'),
    ('Qwen2.5-VL romantic', 'qwen_romantic_150.json', 'romantic'),
    ('Claude humorous', 'claude_humorous.json', 'humorous'),
    ('Claude romantic', 'claude_romantic.json', 'romantic'),
    ('Ours humorous (0.5)', 'gen_humorous_c18k_beam3_off0.5.json', 'humorous'),
    ('Ours romantic (0.4)', 'gen_romantic_c18k_beam3_off0.4.json', 'romantic'),
    # control "Gemini with sight" (evaluation only): same generator and prompt, image attached
    ('Gemini humorous with image', 'gemini_seen_humorous.json', 'humorous'),
    ('Gemini romantic with image', 'gemini_seen_romantic.json', 'romantic'),
    # all five factual captions that Gemini received; their embeddings are averaged per image
    ('Factual, all captions given to Gemini', 'factual_all.json', 'captions'),
]
# factual Stage 1 captioner outputs on the same 1,000 test images (beam search), for factual vs styled comparison
FACTUAL_MODEL = [('Factual model output (Stage 1)', 'pred_factual_model_humorous.json', 'humorous')]
# human-written FlickrStyle10K captions translated into Bangla, only for the 212 test images that have them
HUMAN_SETS = [
    ('Human humorous (FlickrStyle, translated)', 'human_refs_212_final.csv', 'bangla_humorous'),
    ('Human romantic (FlickrStyle, translated)', 'human_refs_212_final.csv', 'bangla_romantic'),
]


def N(t):
    return ' '.join(unicodedata.normalize('NFC', t).split())


def retrieval_metrics(txt, img, gt):
    """txt: (n, d) and img: (m, d) L2-normalised; gt[i] = index of caption i's own image in img."""
    sims = txt @ img.T
    own = sims[np.arange(len(gt)), gt]
    rank = (sims > own[:, None]).sum(1) + 1          # 1 = own image ranked first (ties count in favour)
    return {'n': int(len(gt)), 'R@1': float(100 * np.mean(rank <= 1)), 'R@5': float(100 * np.mean(rank <= 5)),
            'median_rank': float(np.median(rank)), 'CLIP-S': float(np.mean(2.5 * np.clip(own, 0, None)))}


def find(name, roots):
    for r in roots:
        hits = glob.glob(os.path.join(r, '**', name), recursive=True)
        if hits: return hits[0]
    raise SystemExit(f'missing file: {name}')


def load_set(path, field):
    out = {}
    if path.endswith('.csv'):
        for r in csv.DictReader(open(path, encoding='utf-8-sig')):
            if r.get(field, '').strip(): out[r['image_id']] = N(r[field])
    else:
        for r in json.load(open(path, encoding='utf-8')):
            v = r.get(field)
            if isinstance(v, list):
                v = [N(x) for x in v if str(x).strip()]
                if v: out[str(r['image_id'])] = v
            elif str(v or '').strip(): out[str(r['image_id'])] = N(v)
    return out


def encode(caps, keep, enc):
    """One embedding per image. A list of captions is encoded one by one and the normalised mean is used."""
    if not isinstance(caps[keep[0]], list):
        return enc([caps[i] for i in keep]).cpu().numpy()
    flat = [c for i in keep for c in caps[i]]; e = enc(flat).cpu().numpy(); out, k = [], 0
    for i in keep:
        m = e[k:k + len(caps[i])].mean(0); out.append(m / np.linalg.norm(m)); k += len(caps[i])
    return np.stack(out)


def write(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    lines = ['| Caption set | n | R@1 (%) | R@5 (%) | Median rank | CLIP-S |', '|---|---:|---:|---:|---:|---:|']
    for name, m in results.items():
        lines.append(f"| {name} | {m['n']} | {m['R@1']:.1f} | {m['R@5']:.1f} | {m['median_rank']:.0f} | {m['CLIP-S']:.3f} |")
    open(os.path.join(out_dir, 'retrieval.md'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    json.dump(results, open(os.path.join(out_dir, 'retrieval.json'), 'w'), indent=1)
    print('\n'.join(lines), flush=True)


def dummy():
    rng = np.random.default_rng(0); m, d = 1000, 64
    img = rng.normal(size=(m, d)); img /= np.linalg.norm(img, axis=1, keepdims=True)
    res = {}
    for name, noise in (('identical captions', 0.0), ('close captions', 0.03), ('vague captions', 0.15),
                        ('random captions', 1e6)):
        txt = img + noise * rng.normal(size=(m, d)); txt /= np.linalg.norm(txt, axis=1, keepdims=True)
        res[name] = retrieval_metrics(txt, img, np.arange(m))
    res['identical, wrong image ids'] = retrieval_metrics(img, img, np.roll(np.arange(m), 1))
    write(res, 'retrieval_dummy')
    assert res['identical captions']['R@1'] == 100 and res['identical captions']['median_rank'] == 1
    assert res['identical, wrong image ids']['R@1'] == 0
    assert res['close captions']['R@1'] > res['vague captions']['R@1'] > res['random captions']['R@1']
    assert res['random captions']['R@1'] < 2 and all(r['R@5'] >= r['R@1'] for r in res.values())
    print('dummy check passed')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_img'); ap.add_argument('--out', default='retrieval_out')
    ap.add_argument('--roots', nargs='*', default=['/kaggle/input', '.'])
    ap.add_argument('--dummy', action='store_true')
    ap.add_argument('--human212', action='store_true',
                    help='rank only among the 212 test images with human style captions, and add the human rows')
    ap.add_argument('--factual_model_only', action='store_true',
                    help='only the factual Stage 1 output row (other rows are deterministic and already reported)')
    a = ap.parse_args()
    if a.dummy: return dummy()
    import torch
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import evaluate as E
    E.CLIP_NAME = 'nllb-clip-large-siglip'
    test = torch.load(a.test_img, map_location='cpu')
    ids = [str(i) for i in test['ids']]
    img = torch.stack([e.float() for e in test['emb']]); img = (img / img.norm(dim=-1, keepdim=True)).numpy()
    sets = SETS
    if a.human212:
        sub = set(load_set(find('human_refs_212_final.csv', a.roots), 'bangla_humorous'))
        rows = [k for k, i in enumerate(ids) if i in sub]
        ids, img = [ids[k] for k in rows], img[rows]
        sets = HUMAN_SETS + SETS
    if a.factual_model_only:
        sets = FACTUAL_MODEL
    pos = {i: k for k, i in enumerate(ids)}
    print('test images', len(ids), flush=True)
    results = {}
    for name, fname, field in sets:
        caps = load_set(find(fname, a.roots), field)
        keep = [i for i in caps if i in pos]
        txt = encode(caps, keep, E.enc_text)
        results[name] = retrieval_metrics(txt, img, np.array([pos[i] for i in keep]))
        print(name, results[name], flush=True)
    write(results, a.out)


if __name__ == '__main__':
    main()
