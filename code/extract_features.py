# extract_features.py
# NLLB-CLIP SigLIP features: images of each split + styled caption text (for unpaired stage 2).
#   python extract_features.py                          # base model, all, into /kaggle/working/style_feats
#   python extract_features.py --clip large --out /kaggle/working/style_feats_large
#   python extract_features.py --skip_img               # only re-encode styled text (after new captions)
# base  = nllb-clip-base-siglip  (768-d)
# large = nllb-clip-large-siglip (SigLIP ViT-SO400M image encoder, NLLB 1.3B text encoder, 1152-d), Bangla included
# Bangla text is tokenized with bn_text.bn_tokenize (open_clip can no longer set the NLLB language itself).
import argparse
import os, torch
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer
from bn_text import bn_tokenize

ap = argparse.ArgumentParser()
ap.add_argument('--clip', default='base', choices=['base', 'large'])
ap.add_argument('--out', default='/kaggle/working/style_feats')
ap.add_argument('--img_dir', default='/kaggle/input/datasets/kaggleperfect/dataset/data/Images')
ap.add_argument('--skip_img', action='store_true'); ap.add_argument('--skip_txt', action='store_true')
ap.add_argument('--txt_pairs', nargs='*', default=[], metavar='IN:OUT',
                help='encode only these caption files (lines "image_id<TAB>caption" or plain captions), e.g. a.txt:a.pt')
a, _ = ap.parse_known_args()
NAME = {'base': 'nllb-clip-base-siglip', 'large': 'nllb-clip-large-siglip'}[a.clip]
IMG_DIR, OUT = a.img_dir, a.out
BS = 64 if a.clip == 'base' else 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs(OUT, exist_ok=True)
model, preprocess = create_model_from_pretrained(NAME, 'v1', device=device)
tok = get_tokenizer(NAME); model.eval()
print('CLIP', NAME, flush=True)
def strip_ext(n):
    for e in ('.jpg','.jpeg','.png'):
        if n.lower().endswith(e): return n[:-len(e)]
    return n
def image_ids(path):
    seen, order = set(), []
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if not line: continue
        key = line.split('\t',1)[0] if '\t' in line else line.split('#',1)[0]
        img = strip_ext(key.split('#')[0].strip())
        if img not in seen: seen.add(img); order.append(img)
    return order
def read_lines(p): return [l.strip() for l in open(p, encoding='utf-8') if l.strip()]
def amp(): return torch.autocast('cuda', dtype=torch.float16) if device == 'cuda' else torch.autocast('cpu', enabled=False)
@torch.no_grad()
def enc_img(id_file, out):
    ids = image_ids(id_file); feats, kept, batch, bids = [], [], [], []
    def flush():
        if not batch: return
        x = torch.stack(batch).to(device)
        with amp(): f = model.encode_image(x)
        feats.append(f.float().cpu()); kept.extend(bids); batch.clear(); bids.clear()
    for k, img in enumerate(ids):
        path = None
        for e in ('.jpg','.jpeg','.png'):
            p = os.path.join(IMG_DIR, img+e)
            if os.path.exists(p): path = p; break
        if path is None: continue
        batch.append(preprocess(Image.open(path).convert('RGB'))); bids.append(img)
        if len(batch) == BS: flush()
        if k % 5000 == 0: print(' ', os.path.basename(out), k, '/', len(ids), flush=True)
    flush()
    torch.save({'ids': kept, 'emb': torch.cat(feats,0)}, out); print('wrote', out, tuple(feats[0].shape[1:]), flush=True)
@torch.no_grad()
def enc_txt(in_file, out):
    lines = read_lines(in_file) if os.path.exists(in_file) else []; outs = []
    if not lines: print('SKIP (no lines):', in_file, flush=True); return
    for i in range(0, len(lines), BS):
        with amp(): f = model.encode_text(bn_tokenize(tok, lines[i:i+BS]).to(device))
        outs.append(f.float().cpu())
    torch.save({'lines': lines, 'emb': torch.cat(outs,0)}, out); print('wrote', out, flush=True)
if a.txt_pairs:
    for pair in a.txt_pairs:
        src, dst = pair.rsplit(':', 1)
        tmp = dst + '.captions.txt'
        caps = [l.split('\t', 1)[1] if '\t' in l else l for l in read_lines(src)]
        open(tmp, 'w', encoding='utf-8').write('\n'.join(caps) + '\n')
        enc_txt(tmp, dst)
    raise SystemExit(0)
if not a.skip_img:
    enc_img('/kaggle/working/splits/factual_train.txt', f'{OUT}/factual_train_img.pt')
    enc_img('/kaggle/working/splits/factual_val.txt',   f'{OUT}/factual_val_img.pt')
    enc_img('/kaggle/working/splits/factual_test.txt',  f'{OUT}/test_images.pt')
if not a.skip_txt:
    for s in ('romantic', 'humorous'):
        for t in ('train', 'val'):
            enc_txt(f'/kaggle/working/splits/{s}_{t}.txt', f'{OUT}/{s}_{t}.pt')
