# evaluate.py  (v4: all metrics of the paper's evaluation plan, one script for our model and baselines)
#
#   python evaluate.py --style factual  --pred gen_factual.json
#   python evaluate.py --style humorous --pred gen_humorous.json --ckpt sc_paired/paired_humorous_romantic_best.pth
#   python evaluate.py --style humorous --pred qwen_humorous.json --name qwen2.5-vl-7b
#
# Metrics
#   factual only : BLEU-1, BLEU-3, METEOR, ROUGE-L, CIDEr (pycocoevalcap, whitespace tokens, punctuation removed)
#   all styles   : CLIP-S and RefCLIP-S (Hessel et al. 2021, w = 2.5, NLLB-CLIP SigLIP, Bangla text)
#                  Distinct-1/2, Self-BLEU-4
#   styled       : Style accuracy (style_classifier.py, BanglaBERT 3-way: factual / humorous / romantic)
#                  BLEU-1/3, CIDEr vs the image's styled refs (sty) and factual refs (fac), Fact-Sim (grounding)
#   styled+ckpt  : WR-LogP = % of test images where the model gives the reference styled caption a higher
#                  mean per-token log-prob than the reference factual caption (reading of Stylistic
#                  Alignment 2025, arXiv 2509.25568; the paper gives no formula, state this in the paper).
# Results are appended to results.jsonl (one json line per run) for the paper tables.
import os, sys, re, json, argparse, random, unicodedata
import numpy as np, torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ID_KEYS = {'image_id', 'image', 'id', 'img', 'file', 'filename', 'references', 'ref'}
PUNCT = re.compile(r'[।৷॥,.!?;:\'"“”‘’()\[\]{}\-–—…]')

def strip_ext(n):
    for e in ('.jpg', '.jpeg', '.png'):
        if n.lower().endswith(e): return n[:-len(e)]
    return n

def norm(s): return " ".join(PUNCT.sub(" ", unicodedata.normalize("NFC", s)).split())   # NFC as in Mahtab et al. ICCIT 2024

def load_preds(path, style):
    data = json.load(open(path, encoding='utf-8'))
    items = data.values() if isinstance(data, dict) else data
    pairs = []
    for d in items:
        iid = next((str(d[k]) for k in ('image_id', 'image', 'id', 'img') if d.get(k)), None)
        cap = d.get(style) if isinstance(d.get(style), str) else \
              next((v for k, v in d.items() if k not in ID_KEYS and isinstance(v, str) and v.strip()), None)
        if iid and cap and cap.strip(): pairs.append((strip_ext(iid), cap.strip()))
    return pairs

def load_refs(splits, style):
    refs = {}
    fn = f'{splits}/factual_test.txt' if style == 'factual' else f'{splits}/{style}_test_refs.txt'
    for line in open(fn, encoding='utf-8'):
        line = line.strip()
        if not line: continue
        if '\t' in line and style != 'factual': img, cap = line.split('\t', 1)
        else:
            parts = [x.strip() for x in re.split(r'#\d*', line) if x.strip()]
            if len(parts) < 2: continue
            img, cap = parts[0], parts[1]
        if cap.strip(): refs.setdefault(strip_ext(img.strip()), []).append(cap.strip())
    return refs

# ---------------------------------------------------------------- n-gram metrics (factual)
def ngram_metrics(pairs, refs):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge
    gts = {i: [norm(r) for r in refs[i]] for i, _ in pairs if refs.get(i)}
    res = {i: [norm(c)] for i, c in pairs if i in gts}
    out = {}
    b, _ = Bleu(4).compute_score(gts, res, verbose=0)
    for k in (0, 2): out[f'BLEU-{k + 1}'] = 100 * b[k]   # report BLEU-1 and BLEU-3
    out['ROUGE-L'] = 100 * Rouge().compute_score(gts, res)[0]
    out['CIDEr'] = 100 * Cider().compute_score(gts, res)[0]
    try:
        from pycocoevalcap.meteor.meteor import Meteor   # needs Java
        out['METEOR'] = 100 * Meteor().compute_score(gts, res)[0]
    except Exception as e:
        print('  METEOR skipped:', str(e)[:120])
    out['n_images'] = len(gts)
    return out

# ---------------------------------------------------------------- CLIP-S / RefCLIP-S
_clip = None
CLIP_NAME = 'nllb-clip-base-siglip'   # set from --clip; must match the image features used for CLIP-S
def clip_model():
    global _clip
    if _clip is None:
        from open_clip import create_model_from_pretrained, get_tokenizer
        m, _ = create_model_from_pretrained(CLIP_NAME, 'v1', device=device)
        t = get_tokenizer(CLIP_NAME); m.eval(); _clip = (m, t)   # Bangla tag added in bn_tokenize
    return _clip

@torch.no_grad()
def enc_text(caps):
    m, t = clip_model(); out = []
    from bn_text import bn_tokenize
    for s in range(0, len(caps), 64): out.append(m.encode_text(bn_tokenize(t, caps[s:s + 64]).to(device)).float())
    x = torch.cat(out, 0); return x / x.norm(dim=-1, keepdim=True)

def clip_scores(pairs, refs, test_img_pt, w=2.5):
    test = torch.load(test_img_pt, map_location='cpu')
    id2emb = {i: e.float() for i, e in zip(test['ids'], test['emb'])}
    P = [(i, c) for i, c in pairs if i in id2emb]
    img = torch.stack([id2emb[i] for i, _ in P]).to(device); img = img / img.norm(dim=-1, keepdim=True)
    cap = enc_text([c for _, c in P])
    cos = (img * cap).sum(-1)
    if len(P) > 1:   # sanity check: matching pairs must score clearly above mismatched ones
        print(f'  sanity: mean cos matched {cos.mean().item():.4f}  mismatched {(img * cap.roll(1, 0)).sum(-1).mean().item():.4f}')
    clip_s = w * cos.clamp(min=0)
    out = {'CLIP-S': clip_s.mean().item(), 'n_clip': len(P)}
    ref_s = []
    for k, (i, _) in enumerate(P):
        if refs.get(i):
            R = enc_text(refs[i]); ref_s.append(max((R @ cap[k]).max().item(), 0.0))
        else: ref_s.append(None)
    hm = [2 * a * b / (a + b) if b is not None and a + b > 0 else (0.0 if b is not None else None)
          for a, b in zip(clip_s.tolist(), ref_s)]
    hm = [x for x in hm if x is not None]
    if hm: out['RefCLIP-S'] = float(np.mean(hm))
    return out

def fact_sim(pairs, frefs):
    """Grounding: mean of the best CLIP-text cosine between each styled caption and its image's factual captions."""
    P = [(i, c) for i, c in pairs if frefs.get(i)]
    cap = enc_text([c for _, c in P]); sims = []
    for k, (i, _) in enumerate(P):
        sims.append((enc_text(frefs[i]) @ cap[k]).max().item())
    return {'Fact-Sim': float(np.mean(sims)) if sims else float('nan')}

# ---------------------------------------------------------------- diversity
def diversity(caps, n_self=1000, seed=0):
    toks = [norm(c).split() for c in caps]
    out = {}
    for n in (1, 2):
        grams = [tuple(t[k:k + n]) for t in toks for k in range(len(t) - n + 1)]
        out[f'Distinct-{n}'] = len(set(grams)) / max(len(grams), 1)
    from pycocoevalcap.bleu.bleu import Bleu
    rng = random.Random(seed); idx = list(range(len(caps)))
    if len(idx) > n_self: idx = rng.sample(idx, n_self)
    sel = [' '.join(toks[k]) for k in idx]
    gts = {k: sel[:k] + sel[k + 1:] for k in range(len(sel))}
    res = {k: [sel[k]] for k in range(len(sel))}
    out['Self-BLEU-4'] = 100 * Bleu(4).compute_score(gts, res, verbose=0)[0][3]
    return out

# ---------------------------------------------------------------- style accuracy
def style_accuracy(caps, style, clf_dir, max_len=64):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(clf_dir)
    m = AutoModelForSequenceClassification.from_pretrained(clf_dir).to(device).eval()
    labels = ['factual', 'humorous', 'romantic']; pred = []
    with torch.no_grad():
        for s in range(0, len(caps), 64):
            enc = tok(caps[s:s + 64], padding=True, truncation=True, max_length=max_len, return_tensors='pt').to(device)
            pred += m(**enc).logits.argmax(-1).tolist()
    return {'Style-Acc': 100 * sum(p == labels.index(style) for p in pred) / max(len(pred), 1)}

# ---------------------------------------------------------------- WR-LogP (our model)
@torch.no_grad()
def wr_logp(ckpt_path, style, splits, test_img_pt, offset_scale=1.0, max_len=120, bs=16):
    from stage1 import StyleCaptioner, build_tokenizer, add_style_tokens, encode_with_eos, unit
    tok = build_tokenizer(); sid = add_style_tokens(tok)
    ck = torch.load(ckpt_path, map_location='cpu')
    model = StyleCaptioner(ck['clip_dim'], tok, ck['prefix_len']).to(device)
    model.load_state_dict(ck['model']); model.eval()
    offset = ck.get('offset')
    test = torch.load(test_img_pt, map_location='cpu')
    id2emb = {i: e.float() for i, e in zip(test['ids'], test['emb'])}
    sty, fac = load_refs(splits, style), load_refs(splits, 'factual')
    ids = [i for i in sty if i in fac and i in id2emb]

    def mean_logp(emb, caps):
        enc = [encode_with_eos(tok, c, max_len) for c in caps]
        inp = torch.stack([e[0] for e in enc]).to(device); att = torch.stack([e[1] for e in enc]).to(device)
        code = torch.full((len(caps),), sid[style], device=device)
        e = model.embeds(emb, code, inp); P = 1 + model.prefix_len
        mask = torch.cat([torch.ones(len(caps), P, device=device, dtype=att.dtype), att], 1)
        logits = model.gpt(inputs_embeds=e, attention_mask=mask).logits[:, P - 1:-1]
        lp = torch.log_softmax(logits.float(), -1).gather(-1, inp.unsqueeze(-1)).squeeze(-1)
        return ((lp * att).sum(1) / att.sum(1)).tolist()

    wins = 0
    for s in range(0, len(ids), bs):
        ch = ids[s:s + bs]
        emb = unit(torch.stack([id2emb[i] for i in ch]))
        if offset is not None: emb = unit(emb - offset_scale * offset)
        emb = emb.to(device)
        a = mean_logp(emb, [sty[i][0] for i in ch]); b = mean_logp(emb, [fac[i][0] for i in ch])
        wins += sum(x > y for x, y in zip(a, b))
    return {'WR-LogP': 100 * wins / max(len(ids), 1), 'n_wr': len(ids)}

# ---------------------------------------------------------------- main
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--style', required=True, choices=['factual', 'humorous', 'romantic'])
    ap.add_argument('--pred', required=True)
    ap.add_argument('--name', default=None, help='system name in results.jsonl (default: file name)')
    ap.add_argument('--ckpt', default=None, help='our checkpoint, enables WR-LogP')
    ap.add_argument('--offset_scale', type=float, default=1.0, help='same offset_scale as used in generate.py')
    ap.add_argument('--splits', default='/kaggle/working/splits')
    ap.add_argument('--test_img', default='/kaggle/working/style_feats/test_images.pt', help='model input features (WR-LogP)')
    ap.add_argument('--clip', default='base', choices=['base', 'large'], help='CLIP used for CLIP-S (match --clip_img features)')
    ap.add_argument('--clip_img', default=None, help='BASE nllb-clip image features for CLIP-S (default: --test_img)')
    ap.add_argument('--clf', default='/kaggle/working/style_clf')
    ap.add_argument('--results', default='/kaggle/working/results.jsonl')
    a, _ = ap.parse_known_args()
    CLIP_NAME = {'base': 'nllb-clip-base-siglip', 'large': 'nllb-clip-large-siglip'}[a.clip]
    pairs = load_preds(a.pred, a.style)
    print(f'=== evaluate.py v4 === {a.pred}  style {a.style}  captions {len(pairs)}')
    if not pairs: sys.exit('no captions found')
    refs = load_refs(a.splits, a.style)
    R = {'system': a.name or os.path.basename(a.pred), 'style': a.style, 'n': len(pairs)}
    if a.style == 'factual': R.update(ngram_metrics(pairs, refs))
    else:
        # styled captions vs the SAME image's styled references and factual references (content kept = grounding)
        for tag, rr in (('sty', refs), ('fac', load_refs(a.splits, 'factual'))):
            m = ngram_metrics(pairs, rr)
            for k in ('BLEU-1', 'BLEU-3', 'CIDEr'): R[f'{k}({tag})'] = m[k]
        R.update(fact_sim(pairs, load_refs(a.splits, 'factual')))
    R.update(clip_scores(pairs, refs, a.clip_img or a.test_img))   # CLIP-S always with base NLLB-CLIP features
    R.update(diversity([c for _, c in pairs]))
    if a.style != 'factual':
        if os.path.isdir(a.clf): R.update(style_accuracy([c for _, c in pairs], a.style, a.clf))
        else: print('  Style-Acc skipped: train the classifier first (python style_classifier.py train)')
        if a.ckpt: R.update(wr_logp(a.ckpt, a.style, a.splits, a.test_img, a.offset_scale))
    for k, v in R.items(): print(f'  {k:12s} {v:.2f}' if isinstance(v, float) else f'  {k:12s} {v}')
    with open(a.results, 'a', encoding='utf-8') as f: f.write(json.dumps(R, ensure_ascii=False) + '\n')
    print('appended to', a.results)
