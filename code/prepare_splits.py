# prepare_splits.py
# Builds train/val/test splits from the cleaned factual file and the id-keyed styled captions
# (data/styled/part_*.tsv, header image_id<TAB>humorous<TAB>romantic). Styled captions are joined by
# image id, never by line position. Split: random Flickr-only test (N_TEST) and val (N_VAL), rest = train
# (fixed seed, so every run gives the same split). Output file names are the ones extract_features.py,
# stage1.py, stage2.py and generate.py already read.
import os, re, csv, glob, argparse, random

def strip_ext(n):
    for e in ('.jpg', '.jpeg', '.png'):
        if n.lower().endswith(e): return n[:-len(e)]
    return n

def load_styled(styled_dir):
    """All part files (one generator model)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(styled_dir, 'part_*.tsv'))):
        for r in csv.DictReader(open(f, encoding='utf-8'), delimiter='\t'):
            i = r['image_id'].strip()
            if i not in out:
                out[i] = {'humorous': r['humorous'].strip(), 'romantic': r['romantic'].strip()}
    return out

def main(a):
    os.makedirs(a.out, exist_ok=True)
    r = re.compile(r'#\d*')
    order, by_img = [], {}
    for line in open(a.factual, encoding='utf-8'):
        line = line.rstrip('\n')
        if not line.strip(): continue
        parts = [x.strip() for x in r.split(line) if x.strip()]
        if len(parts) < 2: continue
        img = strip_ext(parts[0])
        if img not in by_img: order.append(img); by_img[img] = []
        by_img[img].append(line)
    if a.img_dir and os.path.isdir(a.img_dir):
        order = [i for i in order if any(os.path.exists(os.path.join(a.img_dir, i + e)) for e in ('.jpg', '.jpeg', '.png'))]
    print('images with files:', len(order))

    # test/val: random Flickr images only (5 captions), never collected Bangladeshi images (3 captions),
    # never the first 500 images (part_0001, prompt-example writer), so every test reference is Gemini-written.
    # Fixed seed, so the split is identical on every run.
    part1 = set(order[:a.exclude_first])
    cand = [i for i in order if len(by_img[i]) >= 5 and i not in part1]
    pick = random.Random(a.seed).sample(cand, a.n_test + a.n_val)
    test_set, val_set = set(pick[:a.n_test]), set(pick[a.n_test:])
    test_i = [i for i in order if i in test_set]; val_i = [i for i in order if i in val_set]
    train_i = [i for i in order if i not in test_set and i not in val_set]
    n_bd = lambda L: sum(len(by_img[i]) < 5 for i in L)
    print('train', len(train_i), 'val', len(val_i), 'test', len(test_i),
          ' Bangladeshi in train/val/test:', n_bd(train_i), n_bd(val_i), n_bd(test_i))
    for tag, imgs in [('train', train_i), ('val', val_i), ('test', test_i)]:
        with open(f'{a.out}/factual_{tag}.txt', 'w', encoding='utf-8') as f:
            for i in imgs:
                for ln in by_img[i]: f.write(ln + '\n')

    styled = load_styled(a.styled_dir)
    missing = [i for i in order if i not in styled]
    print('styled captions for', len(order) - len(missing), 'of', len(order), 'images')
    if missing: print('  WARNING: no styled caption yet for', len(missing), 'images, first:', missing[:5])
    for name in ('humorous', 'romantic'):
        for tag, imgs in [('train', train_i), ('val', val_i), ('test', test_i)]:
            rows = [(i, styled[i][name]) for i in imgs if i in styled and styled[i][name]]
            # text only: stage 2 unpaired training (extract_features.enc_txt reads this)
            if tag != 'test':
                open(f'{a.out}/{name}_{tag}.txt', 'w', encoding='utf-8').write('\n'.join(c for _, c in rows) + '\n')
            # id<TAB>caption: test references, and paired stage 2 / style classifier
            fn = f'{a.out}/{name}_test_refs.txt' if tag == 'test' else f'{a.out}/{name}_{tag}_paired.txt'
            open(fn, 'w', encoding='utf-8').write('\n'.join(f'{i}\t{c}' for i, c in rows) + '\n')
            print(name, tag, len(rows))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--factual', default='/kaggle/input/datasets/kaggleperfect/dataset/data/factual_caption.txt')
    p.add_argument('--styled_dir', default='/kaggle/input/datasets/kaggleperfect/dataset/data/styled')
    p.add_argument('--img_dir', default='/kaggle/input/datasets/kaggleperfect/dataset/data/Images')
    p.add_argument('--out', default='/kaggle/working/splits')
    p.add_argument('--n_test', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--exclude_first', type=int, default=500, help='first N images never in val/test')
    p.add_argument('--n_val', type=int, default=2000)
    a, _ = p.parse_known_args(); main(a)
