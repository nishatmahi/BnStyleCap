"""Paired significance, ours vs Qwen2.5-VL 7B (same images): per-caption CIDEr and BERTScore (1000 test images,
Gemini references) and gpt-oss-120b judge ratings (212 images). Paired bootstrap 95% CI of the mean difference
(10,000 resamples) and Wilcoxon signed-rank p-value.  python tools/significance.py
"""
import sys, os, csv, json, unicodedata, collections
sys.modules['sklearn'] = None
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
import numpy as np
from scipy.stats import wilcoxon
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
from diversity_pos import norm
from pycocoevalcap.cider.cider import Cider

rows = {r['image_id']: r for r in csv.DictReader(open(os.path.join(ROOT, 'test_captions_1000 (1).csv'), encoding='utf-8-sig'))}
ids = list(rows)
rng = np.random.default_rng(0)


def stats(a, b):
    d = np.asarray(a) - np.asarray(b)
    boot = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    p = wilcoxon(a, b).pvalue if np.any(d != 0) else 1.0
    return d.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5), p


out = ['| Metric | Style | Ours | Qwen | Diff (Ours - Qwen) | 95% CI | Wilcoxon p | n |', '|---|---|---|---|---|---|---|---|']
def add(name, st, a, b):
    m, lo, hi, p = stats(a, b)
    out.append(f'| {name} | {st} | {np.mean(a):.3f} | {np.mean(b):.3f} | {m:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {p:.1e} | {len(a)} |')
    print(out[-1], flush=True)

per = {}
for st in ('humorous', 'romantic'):
    q = {x['image_id']: x[st] for x in json.load(open(os.path.join(ROOT, f'qwen_{st}_150.json'), encoding='utf-8'))}
    gts = {i: [norm(rows[i][st + '_reference'])] for i in ids}
    c_o = Cider().compute_score(gts, {i: [norm(rows[i][st + '_generated'])] for i in ids})[1]
    c_q = Cider().compute_score(gts, {i: [norm(q[i])] for i in ids})[1]
    add('CIDEr (x100)', st, 100 * np.array(c_o), 100 * np.array(c_q))
    per[st] = (q, gts)

import transformers
_orig = transformers.AutoTokenizer.from_pretrained
def _tok(*a, **k):
    t = _orig(*a, **k); t.model_max_length = 512; return t
transformers.AutoTokenizer.from_pretrained = _tok
from bert_score import score
nfc = lambda s: ' '.join(unicodedata.normalize('NFC', s).split())
for st in ('humorous', 'romantic'):
    q, _ = per[st]
    refs = [nfc(rows[i][st + '_reference']) for i in ids]
    Fo = score([nfc(rows[i][st + '_generated']) for i in ids], refs, model_type='csebuetnlp/banglabert', num_layers=9, batch_size=32)[2]
    Fq = score([nfc(q[i]) for i in ids], refs, model_type='csebuetnlp/banglabert', num_layers=9, batch_size=32)[2]
    add('BERTScore F1 (x100)', st, 100 * Fo.numpy(), 100 * Fq.numpy())

J = collections.defaultdict(dict)
for o in map(json.loads, open(os.path.join(ROOT, 'judge_gptoss', 'judge_scores_final.jsonl'), encoding='utf-8')):
    J[(o['system'], o['style'])][o['id']] = o
for st in ('humorous', 'romantic'):
    common = sorted(set(J[('ours', st)]) & set(J[('qwen25vl_7b', st)]))
    for k, name in (('style_score', 'Judge style'), ('grounding', 'Judge content consistency'), ('fluency', 'Judge fluency')):
        add(name, st, [J[('ours', st)][i][k] for i in common], [J[('qwen25vl_7b', st)][i][k] for i in common])

open(os.path.join(ROOT, 'significance_results.md'), 'w', encoding='utf-8').write('\n'.join(out) + '\n')
print('saved significance_results.md')
