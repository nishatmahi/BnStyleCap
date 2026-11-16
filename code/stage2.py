import os, random, time
from collections import Counter
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import argparse, torch, torch.nn as nn
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from transformers import AutoTokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
from stage1 import (StyleCaptioner, build_tokenizer, add_style_tokens, read_lines,
                    split_line, encode_with_eos, unit)

class FactualImgData(Dataset):
    def __init__(self, img_pt, cap_file, tok, sid, limit=None, max_len=120):
        d = torch.load(img_pt, map_location='cpu')
        emb_by_id = {i: e.float() for i,e in zip(d['ids'], d['emb'])}
        rows = []
        for ln in read_lines(cap_file):
            p = split_line(ln)
            if not p or not p[1]: continue
            img, cap = p
            if img in emb_by_id: rows.append((emb_by_id[img], cap))
        if limit and len(rows) > limit:
            random.seed(0); random.shuffle(rows); rows = rows[:limit]
        self.rows, self.tok, self.code, self.max_len = rows, tok, sid['factual'], max_len
        print('[data] factual', len(rows),'rows', flush=True)
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        emb, cap = self.rows[i]
        ids, attn = encode_with_eos(self.tok, cap, self.max_len)
        return unit(emb), torch.tensor(self.code), ids, attn

def template_keep(lines, end_min=100, tri_pct=2.0):
    """Indices of captions without a template: ending (last 2 words) used >= end_min times,
    or any word 3-gram that occurs in >= tri_pct % of the captions."""
    words = [l.split() for l in lines]
    ends = Counter(' '.join(w[-2:]) for w in words)
    grams = [{' '.join(w[i:i + 3]) for i in range(len(w) - 2)} for w in words]
    df = Counter(g for gs in grams for g in gs)
    bad = {g for g, n in df.items() if n >= tri_pct / 100 * len(lines)}
    return [i for i, w in enumerate(words) if ends[' '.join(w[-2:])] < end_min and not (grams[i] & bad)]

class StyleTextData(Dataset):
    def __init__(self, pt, tok, sid, style, noise=0.05, max_len=120, limit=None, filter_templates=False):
        d = torch.load(pt, map_location='cpu')
        self.emb = d['emb'].float(); self.lines = d['lines']
        if filter_templates:
            keep = template_keep(self.lines)
            print(f'[data] template filter keeps {len(keep)} of {len(self.lines)}', flush=True)
            self.emb = self.emb[keep]; self.lines = [self.lines[i] for i in keep]
        if limit and len(self.lines) > limit:
            keep = sorted(random.Random(0).sample(range(len(self.lines)), limit))
            self.emb = self.emb[keep]; self.lines = [self.lines[i] for i in keep]
        self.tok, self.code, self.noise, self.max_len = tok, sid[style], noise, max_len
        print('[data]', style, len(self.lines),'rows', flush=True)
    def __len__(self): return len(self.lines)
    def __getitem__(self, i):
        e = unit(self.emb[i])
        if self.noise > 0: e = e + torch.randn_like(e)*self.noise
        ids, attn = encode_with_eos(self.tok, self.lines[i], self.max_len)
        return e, torch.tensor(self.code), ids, attn

def norm_mean(pt, key='emb'):
    d = torch.load(pt, map_location='cpu')
    return unit(d[key].float()).mean(0)

@torch.no_grad()
def eval_loss(model, dl, dev):
    model.eval(); s = n = 0
    for emb, code, ids, attn in dl:
        s += model(emb.to(dev), code.to(dev), ids.to(dev), attn.to(dev)).item(); n += 1
    return s / max(n,1)

def main(a):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(a.save_dir, exist_ok=True)
    tok = build_tokenizer(); sid = add_style_tokens(tok)
    init = torch.load(a.init, map_location='cpu')
    model = StyleCaptioner(init['clip_dim'], tok, init['prefix_len']).to(dev)
    model.load_state_dict(init['model'])
    sty  = StyleTextData(a.style_pt,     tok, sid, a.style, noise=a.noise, max_len=a.max_len, limit=a.style_limit,
                         filter_templates=a.filter_templates)
    styv = StyleTextData(a.style_val_pt, tok, sid, a.style, noise=0.0, max_len=a.max_len)
    if a.no_factual:
        print('[data] no factual image-caption pairs in stage 2 (ablation)', flush=True)
        train_sets, val_sets = [sty], [styv]
    else:
        fac  = FactualImgData(a.factual_img, a.factual_cap, tok, sid, limit=len(sty), max_len=a.max_len)
        facv = FactualImgData(a.factual_val_img, a.factual_val_cap, tok, sid, limit=len(styv), max_len=a.max_len)
        train_sets, val_sets = [fac, sty], [facv, styv]
    dl   = DataLoader(ConcatDataset(train_sets), batch_size=a.batch_size, shuffle=True, num_workers=2)
    va_dl = DataLoader(ConcatDataset(val_sets), batch_size=a.batch_size, shuffle=False, num_workers=2)
    offset = (norm_mean(a.factual_img) - unit(sty.emb).mean(0)).to(dev)   # same captions the run trains on
    print('modality offset norm', round(offset.norm().item(),4), flush=True)
    print('offset_scale', a.offset_scale, flush=True)
    timg = torch.load(a.test_img, map_location='cpu')
    temb = {i: e.float() for i,e in zip(timg['ids'], timg['emb'])}
    order, refs = [], {}
    for ln in read_lines(a.test_cap):
        p = split_line(ln)
        if not p or not p[1]: continue
        img, cap = p
        if img in temb and img not in refs: refs[img] = cap; order.append(img)
        if len(order) >= 6: break
    peek = unit(torch.stack([temb[i] for i in order]))
    peek = unit(peek - a.offset_scale * offset.cpu()).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, 200, a.epochs*len(dl))
    ckpt = os.path.join(a.save_dir, f'{a.style}.pth')
    best_ckpt = os.path.join(a.save_dir, f'{a.style}_best.pth')
    start_epoch = 0; best = 1e9
    if os.path.exists(ckpt) and not a.fresh:
        r = torch.load(ckpt, map_location='cpu')
        model.load_state_dict(r['model'])
        if 'opt' in r: opt.load_state_dict(r['opt'])
        if 'sch' in r: sch.load_state_dict(r['sch'])
        start_epoch = r.get('epoch',0); best = r.get('best',1e9)
        print('RESUMING from epoch', start_epoch,' best val', round(best,4), flush=True)
    else:
        print('starting fresh from', a.init, flush=True)
    for ep in range(start_epoch, a.epochs):
        t_ep = time.time()
        model.train(); tr_sum = tr_n = 0
        for i,(emb, code, ids, attn) in enumerate(dl):
            emb, code, ids, attn = emb.to(dev), code.to(dev), ids.to(dev), attn.to(dev)
            loss = model(emb, code, ids, attn); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad()
            tr_sum += loss.item(); tr_n += 1
            if i % 500 == 0: print('ep', ep+1, i,'/',len(dl),'loss',round(loss.item(),3), flush=True)
        tr_loss = tr_sum/max(tr_n,1); val = eval_loss(model, va_dl, dev)
        print('[EPOCH', ep+1,'] train_loss', round(tr_loss,4),' val_loss', round(val,4), flush=True)
        model.eval()
        code = torch.full((peek.size(0),), sid[a.style], device=dev)
        if a.beams > 1:
            outs = model.generate_beam(peek, code, max_new=a.max_new, eos_id=tok.eos_token_id, beams=a.beams)
        else:
            outs = model.generate(peek, code, max_new=a.max_new, eos_id=tok.eos_token_id)
        for iid, o in zip(order, outs):
            print(' ', iid, a.style,' ', tok.decode(o, skip_special_tokens=True))
            print('      factual ref:', refs[iid])
        meta = {'model': model.state_dict(),'opt': opt.state_dict(),'sch': sch.state_dict(),
                'epoch': ep+1,'best': best,'clip_dim': init['clip_dim'],
                'prefix_len': init['prefix_len'],'offset': offset.cpu(),'style': a.style}
        if val < best:
            best = val; meta['best'] = best
            torch.save(meta, best_ckpt); print('  NEW BEST -> saved', os.path.basename(best_ckpt), flush=True)
        else:
            print('  val did NOT improve (best', round(best,4),')', flush=True)
        torch.save(meta, ckpt)
        # small model-only copy of every epoch, so the epoch can be chosen afterwards (peek + quick val check)
        torch.save({k: meta[k] for k in ('model', 'epoch', 'clip_dim', 'prefix_len', 'offset', 'style')},
                   os.path.join(a.save_dir, f'{a.style}_ep{ep+1}.pth'))
        print('  saved', f'{a.style}_ep{ep+1}.pth', flush=True)
        if a.stop_at and ep + 1 < a.epochs and time.time() + (time.time() - t_ep) > a.stop_at:
            print(f'TIME GUARD: stopping after epoch {ep+1} (next epoch would pass the deadline); rerun to resume', flush=True)
            break
    print('done', flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--style', default='romantic', choices=['romantic','humorous'])
    p.add_argument('--init',            default='/kaggle/working/sc_factual/best.pth')
    p.add_argument('--factual_img',     default='/kaggle/working/style_feats/factual_train_img.pt')
    p.add_argument('--factual_cap',     default='/kaggle/working/splits/factual_train.txt')
    p.add_argument('--factual_val_img', default='/kaggle/working/style_feats/factual_val_img.pt')
    p.add_argument('--factual_val_cap', default='/kaggle/working/splits/factual_val.txt')
    p.add_argument('--style_pt',        default=None)
    p.add_argument('--style_val_pt',    default=None)
    p.add_argument('--test_img',        default='/kaggle/working/style_feats/test_images.pt')
    p.add_argument('--test_cap',        default='/kaggle/working/splits/factual_test.txt')
    p.add_argument('--save_dir',        default=None)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--noise', type=float, default=0.05)
    p.add_argument('--max_len', type=int, default=120)
    p.add_argument('--max_new', type=int, default=120)
    p.add_argument('--offset_scale', type=float, default=1.0)
    p.add_argument('--beams', type=int, default=1)
    p.add_argument('--fresh', action='store_true')
    p.add_argument('--no_factual', action='store_true', help='ablation: stage 2 on style text only')
    p.add_argument('--stop_at', type=float, default=0, help='unix time: stop cleanly before an epoch that would end after it, 0 = off')
    p.add_argument('--style_limit', type=int, default=0, help='random subset of N style captions (seed 0), 0 = all')
    p.add_argument('--filter_templates', action='store_true',
                   help='drop templated training captions (frequent ending or frequent 3-gram) before --style_limit')
    a, _ = p.parse_known_args()
    if a.style_pt     is None: a.style_pt     = f'/kaggle/working/style_feats/{a.style}_train.pt'
    if a.style_val_pt is None: a.style_val_pt = f'/kaggle/working/style_feats/{a.style}_val.pt'
    if a.save_dir     is None: a.save_dir     = f'/kaggle/working/sc_{a.style}'
    main(a)
