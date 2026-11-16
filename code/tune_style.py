# tune_style.py
# Picks the styled-generation setting on VALIDATION images (never test), so the paper can report tuned values.
# For each setting: generate captions for N val images, score Style-Acc (style_clf) and CLIP-S (image match).
# Best = highest Style-Acc among settings whose CLIP-S stays >= keep * CLIP-S of the plain offset baseline.
#
#   python tune_style.py --style humorous --ckpt /kaggle/working/sc_humorous/humorous_best.pth
# Needs: style_feats/factual_val_img.pt, splits/factual_val.txt, style_feats/<style>_train.pt, trained style_clf.
import os, sys, json, argparse, subprocess, itertools
import evaluate as E

def run(a, name, extra):
    out = f'{a.work}/tune_{a.style}_{name}.json'
    cmd = [sys.executable, 'generate.py', '--ckpt', a.ckpt, '--style', a.style, '--out_json', out,
           '--test_img', a.val_img, '--test_cap', a.val_cap, '--n_images', str(a.n), '--beams', str(a.beams)] + extra
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    pairs = E.load_preds(out, a.style)
    r = {'setting': name, 'n': len(pairs)}
    r.update(E.clip_scores(pairs, {}, a.val_img))
    r.update(E.style_accuracy([c for _, c in pairs], a.style, a.clf))
    print(f"  {name:38s} Style-Acc {r['Style-Acc']:6.2f}   CLIP-S {r['CLIP-S']:.4f}", flush=True)
    return r

def main(a):
    E.CLIP_NAME = {'base': 'nllb-clip-base-siglip', 'large': 'nllb-clip-large-siglip'}[a.clip]   # must match the val image features
    os.makedirs(a.work, exist_ok=True)
    mem = f'/kaggle/working/style_feats/{a.style}_train.pt'
    grid = [('offset_%s' % s, ['--offset_scale', str(s)]) for s in a.offsets]
    grid += [('proj_t%s' % t, ['--proj_mem', mem, '--proj_temp', str(t)]) for t in a.temps]
    base = [g for g in grid if g[0] == 'offset_%s' % a.offsets[0]][0]
    for (gn, gx), lam in itertools.product([base] + [g for g in grid if g[0].startswith('proj')], a.lams):
        grid.append((f'{gn}+cfg{lam}', gx + ['--guidance', str(lam)]))
    print(f'tuning {a.style} on {a.n} val images, {len(grid)} settings', flush=True)
    res = [run(a, n, x) for n, x in grid]
    b0 = next(r for r in res if r['setting'] == base[0])
    ok = [r for r in res if r['CLIP-S'] >= a.keep * b0['CLIP-S']]
    best = max(ok, key=lambda r: r['Style-Acc'])
    print(f"\nbaseline {b0['setting']}: Style-Acc {b0['Style-Acc']:.2f} CLIP-S {b0['CLIP-S']:.4f}")
    print(f"BEST {best['setting']}: Style-Acc {best['Style-Acc']:.2f} CLIP-S {best['CLIP-S']:.4f}  (CLIP-S kept >= {a.keep:.0%} of baseline)")
    json.dump({'results': res, 'best': best}, open(f'{a.work}/tune_{a.style}.json', 'w'), indent=2)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--style', required=True, choices=['humorous', 'romantic'])
    p.add_argument('--ckpt', required=True)
    p.add_argument('--val_img', default='/kaggle/working/style_feats/factual_val_img.pt')
    p.add_argument('--val_cap', default='/kaggle/working/splits/factual_val.txt')
    p.add_argument('--clf', default='/kaggle/working/style_clf')
    p.add_argument('--work', default='/kaggle/working/tune')
    p.add_argument('--n', type=int, default=200)
    p.add_argument('--beams', type=int, default=3)
    p.add_argument('--offsets', type=float, nargs='+', default=[0.5, 0.3, 1.0])
    p.add_argument('--temps', type=float, nargs='*', default=[0.01, 0.05])
    p.add_argument('--lams', type=float, nargs='*', default=[0.5, 1.0, 2.0])
    p.add_argument('--keep', type=float, default=0.9)
    p.add_argument('--clip', default='large', choices=['base', 'large'])
    a, _ = p.parse_known_args(); main(a)
