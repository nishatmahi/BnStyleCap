# style_classifier.py
# 3-way Bangla style classifier (factual / humorous / romantic) used for the Style Accuracy metric.
# Train:   python style_classifier.py train
#          -> trains on train-split captions, picks best epoch on val, reports its own
#             precision / recall / F1 per class on the held-out test references (report this in the paper).
# Report:  python style_classifier.py report   -> test P/R/F1 of the saved classifier, no training
# Score:   python style_classifier.py score --pred /kaggle/working/gen_humorous.json --style humorous
#          -> style accuracy = share of generated captions classified as the target style.
#             Works for our generate.py output and for the baseline outputs (same json format:
#             list of {image_id, <style>: caption}).
import os, re, json, random, argparse
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

LABELS = ['factual', 'humorous', 'romantic']

def split_line(line):
    if '\t' in line: k, c = line.split('\t', 1)
    elif '#' in line:
        k, c = line.split('#', 1); parts = c.split(None, 1); c = parts[1] if len(parts) > 1 else ''
    else: return None
    k = k.split('#')[0].strip()
    for e in ('.jpg', '.jpeg', '.png'):
        if k.lower().endswith(e): k = k[:-len(e)]
    return k, c.strip()

def read_pairs(p):
    out = []
    for l in open(p, encoding='utf-8'):
        x = split_line(l.strip()) if l.strip() else None
        if x and x[1]: out.append(x)
    return out

def load_split(sp, tag, seed=0):
    """One factual caption per image (random) so the 3 classes are balanced."""
    rng = random.Random(seed)
    fac = {}
    for i, c in read_pairs(f'{sp}/factual_{tag}.txt'): fac.setdefault(i, []).append(c)
    rows = [(rng.choice(cs), 0) for cs in fac.values()]
    for k, s in ((1, 'humorous'), (2, 'romantic')):
        fn = f'{sp}/{s}_test_refs.txt' if tag == 'test' else f'{sp}/{s}_{tag}_paired.txt'
        rows += [(c, k) for _, c in read_pairs(fn)]
    return rows

def batches(tok, rows, bs, shuffle, max_len):
    def coll(b):
        enc = tok([t for t, _ in b], padding=True, truncation=True, max_length=max_len, return_tensors='pt')
        enc['labels'] = torch.tensor([y for _, y in b]); return enc
    return DataLoader(rows, batch_size=bs, shuffle=shuffle, collate_fn=coll)

@torch.no_grad()
def predict(model, tok, texts, dev, bs=64, max_len=64):
    model.eval(); out = []
    for s in range(0, len(texts), bs):
        enc = tok(texts[s:s + bs], padding=True, truncation=True, max_length=max_len, return_tensors='pt').to(dev)
        out += model(**enc).logits.argmax(-1).tolist()
    return out

def prf(gold, pred):
    rep = {}
    for k, name in enumerate(LABELS):
        tp = sum(g == k and p == k for g, p in zip(gold, pred))
        fp = sum(g != k and p == k for g, p in zip(gold, pred))
        fn = sum(g == k and p != k for g, p in zip(gold, pred))
        P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
        rep[name] = {'precision': P, 'recall': R, 'f1': 2 * P * R / (P + R) if P + R else 0.0, 'support': tp + fn}
    rep['accuracy'] = sum(g == p for g, p in zip(gold, pred)) / max(len(gold), 1)
    rep['macro_f1'] = sum(rep[n]['f1'] for n in LABELS) / len(LABELS)
    return rep

def train(a):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(a.seed); random.seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=3).to(dev)
    tr, va, te = (load_split(a.splits, t, a.seed) for t in ('train', 'val', 'test'))
    print('rows train', len(tr), 'val', len(va), 'test', len(te), flush=True)
    dl = batches(tok, tr, a.batch_size, True, a.max_len)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * a.epochs * len(dl)), a.epochs * len(dl))
    os.makedirs(a.out, exist_ok=True); best = -1
    for ep in range(a.epochs):
        model.train()
        for i, b in enumerate(dl):
            loss = model(**{k: v.to(dev) for k, v in b.items()}).loss; loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad()
            if i % 500 == 0: print('ep', ep + 1, i, '/', len(dl), 'loss', round(loss.item(), 4), flush=True)
        r = prf([y for _, y in va], predict(model, tok, [t for t, _ in va], dev, max_len=a.max_len))
        print(f'[EPOCH {ep + 1}] val acc {r["accuracy"]:.4f} macro-F1 {r["macro_f1"]:.4f}', flush=True)
        if r['macro_f1'] > best:
            best = r['macro_f1']; model.save_pretrained(a.out); tok.save_pretrained(a.out)
            print('  NEW BEST -> saved', a.out, flush=True)
    report(a, tok, te)

def report(a, tok=None, te=None):
    """Test precision / recall / F1 of the SAVED classifier (no training)."""
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok = tok or AutoTokenizer.from_pretrained(a.out)
    te = te or load_split(a.splits, 'test', a.seed)
    model = AutoModelForSequenceClassification.from_pretrained(a.out).to(dev)
    r = prf([y for _, y in te], predict(model, tok, [t for t, _ in te], dev, max_len=a.max_len))
    json.dump(r, open(os.path.join(a.out, 'test_report.json'), 'w'), indent=2)
    print('\nCLASSIFIER TEST REPORT (held-out test references)')
    for n in LABELS: print(f'  {n:9s} P {r[n]["precision"]:.4f}  R {r[n]["recall"]:.4f}  F1 {r[n]["f1"]:.4f}  n {r[n]["support"]}')
    print(f'  accuracy {r["accuracy"]:.4f}  macro-F1 {r["macro_f1"]:.4f}')

def score(a):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok = AutoTokenizer.from_pretrained(a.out)
    model = AutoModelForSequenceClassification.from_pretrained(a.out).to(dev)
    rows = json.load(open(a.pred, encoding='utf-8'))
    texts = [r[a.style] for r in rows if r.get(a.style)]
    pred = predict(model, tok, texts, dev, max_len=a.max_len)
    k = LABELS.index(a.style)
    dist = {n: sum(p == j for p in pred) / max(len(pred), 1) for j, n in enumerate(LABELS)}
    acc = dist[a.style]
    print(f'{a.pred}: style accuracy ({a.style}) {acc:.4f} on {len(pred)} captions   predicted mix',
          {n: round(v, 4) for n, v in dist.items()})
    return acc

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['train', 'score', 'report'])
    p.add_argument('--model', default='csebuetnlp/banglabert')
    p.add_argument('--splits', default='/kaggle/working/splits')
    p.add_argument('--out', default='/kaggle/working/style_clf')
    p.add_argument('--pred', default=None)
    p.add_argument('--style', default='humorous', choices=LABELS)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--max_len', type=int, default=64)
    p.add_argument('--seed', type=int, default=0)
    a, _ = p.parse_known_args()
    {'train': train, 'score': score, 'report': report}[a.cmd](a)
