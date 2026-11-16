"""Independent style classifier data (LT-StyleCap protocol: classifier trained independently on FlickrStyle10K).

Humorous / romantic = human-written FlickrStyle10K captions (public 7k, uberger/re_cap files in human_refs/),
machine-translated to Bangla with NLLB-200 (no Gemini text anywhere). Factual = the same images' original English
Flickr30k factual caption, translated by the same NLLB model, so all three classes share one translator. Images in our 1000-image test split are excluded.

  python tools/build_fs_classifier_data.py prepare     # writes human_refs/fs_clf/en_source.tsv and the image split
  python tools/build_fs_classifier_data.py translate   # NLLB-200 en->bn, resumable, writes bn_translated.tsv
  python tools/build_fs_classifier_data.py splits      # writes human_refs/fs_clf/splits/ in style_classifier.py format
Then: python style_classifier.py train --splits <that folder> --out <new folder>
"""
import os, re, sys, csv, json, random

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
HR = os.path.join(ROOT, 'human_refs'); OUT = os.path.join(HR, 'fs_clf'); os.makedirs(OUT, exist_ok=True)
SRC, BN, SPLIT = (os.path.join(OUT, n) for n in ('en_source.tsv', 'bn_translated_google.tsv', 'image_split.json'))


def load_fs():
    d = {}
    for st, files in (('humorous', ('humor_train', 'humor_test')), ('romantic', ('romantic_train', 'romantic_test'))):
        for f in files:
            for x in json.load(open(os.path.join(HR, f + '.json'), encoding='utf-8')):
                e = d.setdefault(x['image'].split('/')[-1].split('.')[0], {})
                e[st] = x['answer'].strip()
                e.setdefault('factual', x['question'].strip())   # original English Flickr30k factual caption
    return {i: v for i, v in d.items() if 'humorous' in v and 'romantic' in v}


def prepare():
    fs = load_fs()
    test = {r['image_id'] for r in csv.DictReader(open(os.path.join(ROOT, 'test_captions_1000 (1).csv'), encoding='utf-8-sig'))}
    ids = sorted(i for i in fs if i not in test)
    random.Random(0).shuffle(ids)
    split = {'test': ids[:500], 'val': ids[500:1000], 'train': ids[1000:]}
    json.dump(split, open(SPLIT, 'w'), indent=0)
    with open(SRC, 'w', encoding='utf-8') as f:
        for i in ids:
            for st in ('humorous', 'romantic'):
                f.write(f'{i}\t{st}\t{fs[i][st]}\n')
    print('images', len(ids), {k: len(v) for k, v in split.items()}, '| excluded test overlap', len(fs) - len(ids))


def translate(batch=48):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    name = 'facebook/nllb-200-distilled-600M'
    tok = AutoTokenizer.from_pretrained(name, src_lang='eng_Latn'); model = AutoModelForSeq2SeqLM.from_pretrained(name).eval()
    torch.set_num_threads(max(1, os.cpu_count() - 1))
    rows = [l.rstrip('\n').split('\t') for l in open(SRC, encoding='utf-8')]
    done = set()
    if os.path.exists(BN):
        done = {tuple(l.split('\t')[:2]) for l in open(BN, encoding='utf-8')}
    todo = [r for r in rows if (r[0], r[1]) not in done]
    print('to translate', len(todo), flush=True)
    bos = tok.convert_tokens_to_ids('ben_Beng')
    with open(BN, 'a', encoding='utf-8') as f, torch.no_grad():
        for s in range(0, len(todo), batch):
            ch = todo[s:s + batch]
            enc = tok([re.sub(r'\s+([.,!?])', r'\1', r[2]) for r in ch], return_tensors='pt', padding=True, truncation=True, max_length=64)
            out = model.generate(**enc, forced_bos_token_id=bos, num_beams=1, max_new_tokens=64)
            for r, o in zip(ch, tok.batch_decode(out, skip_special_tokens=True)):
                f.write(f'{r[0]}\t{r[1]}\t{o.strip()}\n')
            f.flush()
            if s % (batch * 25) == 0: print(f'  {s + len(ch)}/{len(todo)}', flush=True)


def splits():
    split = json.load(open(SPLIT))
    bn = {}
    for l in open(BN, encoding='utf-8'):
        i, st, c = l.rstrip('\n').split('\t'); bn[(i, st)] = c
    fac = {}
    for l in open(os.path.join(ROOT, 'data', 'factual_caption.txt'), encoding='utf-8'):
        p = [x.strip() for x in re.split(r'#\d*', l.strip()) if x.strip()]
        if len(p) >= 2: fac.setdefault(re.sub(r'\.(jpg|jpeg|png)$', '', p[0], flags=re.I), []).append(p[1])
    sp = os.path.join(OUT, 'splits'); os.makedirs(sp, exist_ok=True)
    for tag, ids in split.items():
        ids = [i for i in ids if i in fac and all((i, s) in bn for s in ('humorous', 'romantic'))]
        with open(os.path.join(sp, f'factual_{tag}.txt'), 'w', encoding='utf-8') as f:
            for i in ids:
                for c in fac[i]: f.write(f'{i}\t{c}\n')
        for st in ('humorous', 'romantic'):
            fn = f'{st}_test_refs.txt' if tag == 'test' else f'{st}_{tag}_paired.txt'
            with open(os.path.join(sp, fn), 'w', encoding='utf-8') as f:
                for i in ids: f.write(f'{i}\t{bn[(i, st)]}\n')
        print(tag, len(ids), 'images')


if __name__ == '__main__':
    {'prepare': prepare, 'translate': translate, 'splits': splits}[sys.argv[1]]()
