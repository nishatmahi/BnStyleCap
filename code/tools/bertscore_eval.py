"""BERTScore (Zhang et al., ICLR 2020) with csebuetnlp/banglabert for the styled captions.
Systems: factual model, Qwen2.5-VL 7B (150-token run), ours. References: Gemini (1000 test images) and the
212 human FlickrStyle references. Layer 9 = bert-score's default for ELECTRA-base models (BanglaBERT is ELECTRA-base).
Raw F1 (no baseline rescaling: bert-score has no Bangla baseline file).
  python tools/bertscore_eval.py
"""
import sys, os, csv, json, unicodedata
sys.modules['sklearn'] = None            # Windows Application Control blocks a sklearn DLL, not needed here
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
import transformers
_orig = transformers.AutoTokenizer.from_pretrained
def _tok(*a, **k):                        # BanglaBERT's tokenizer reports no length limit, bert-score then overflows
    t = _orig(*a, **k); t.model_max_length = 512; return t
transformers.AutoTokenizer.from_pretrained = _tok
from bert_score import score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
nfc = lambda s: ' '.join(unicodedata.normalize('NFC', s).split())
rows = {r['image_id']: r for r in csv.DictReader(open(os.path.join(ROOT, 'test_captions_1000 (1).csv'), encoding='utf-8-sig'))}
hum = {r['image_id']: r for r in csv.DictReader(open(os.path.join(ROOT, 'human_refs', 'human_refs_212_final.csv'), encoding='utf-8-sig'))}

out, lines = {}, ['| Style | System | BERTScore F1 vs Gemini refs (1000) | BERTScore F1 vs human refs (212) |', '|---|---|---|---|']
for st, hc in (('humorous', 'bangla_humorous'), ('romantic', 'bangla_romantic')):
    q = {x['image_id']: x[st] for x in json.load(open(os.path.join(ROOT, f'qwen_{st}_150.json'), encoding='utf-8'))}
    systems = {'Factual model': {i: r['factual_generated'] for i, r in rows.items()}, 'Qwen2.5-VL 7B': q,
               'Ours': {i: r[st + '_generated'] for i, r in rows.items()}}
    for name, s in systems.items():
        res = []
        for ids, ref in ((list(rows), lambda i: rows[i][st + '_reference']), (list(hum), lambda i: hum[i][hc])):
            P, R, F = score([nfc(s[i]) for i in ids], [nfc(ref(i)) for i in ids], model_type='csebuetnlp/banglabert',
                            num_layers=9, batch_size=32, lang='bn')
            res.append(round(100 * F.mean().item(), 2))
        out[f'{st}|{name}'] = res
        lines.append(f'| {st} | {name} | {res[0]:.2f} | {res[1]:.2f} |'); print(lines[-1], flush=True)
open(os.path.join(ROOT, 'bertscore_results.md'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
json.dump(out, open(os.path.join(ROOT, 'bertscore_results.json'), 'w'), indent=1)
print('saved bertscore_results.md')
