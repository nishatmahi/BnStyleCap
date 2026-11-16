# generate.py
# Default behaviour is unchanged. Two optional inference-time improvements for styled captions
# (no retraining, work with existing stage-2 checkpoints):
#   --proj_mem <style_train.pt>  projection (DeCap, ICLR 2023): the image embedding is replaced by a
#                                similarity-weighted mix of style TEXT embeddings from training, so the decoder
#                                sees inputs like the ones it was trained on (instead of image minus mean offset)
#   --guidance <lambda>          style guidance (classifier-free guidance): at each step
#                                logp = (1 + lambda) * logp(style) - lambda * logp(factual), pushes toward the style
import os, json, math, argparse, torch
from stage1 import (StyleCaptioner, build_tokenizer, add_style_tokens,
                    read_lines, split_line, unit, banned_ngram)

def project(e, mem, temp):
    """DeCap projection onto the memory of style text embeddings (mem: unit rows, on e's device)."""
    w = torch.softmax((e @ mem.T) / temp, dim=-1)
    return unit(w @ mem)

@torch.no_grad()
def guided_decode(model, e_s, e_f, c_s, c_f, lam, beams, max_new, eos_id, rep, len_penalty, no_repeat):
    """Beam search (beams=1 is greedy) with style guidance: style stream on e_s, factual stream on e_f."""
    wte = model.gpt.transformer.wte; w = model.clip_project.net[0].weight
    pref = lambda e, c: torch.cat([wte(c).unsqueeze(1), model.clip_project(e.to(w.dtype))], dim=1)
    P_s, P_f = pref(e_s, c_s), pref(e_f, c_f)
    dev = P_s.device; log_rep = math.log(rep); results = []
    for b in range(P_s.size(0)):
        ps, pf = P_s[b:b+1], P_f[b:b+1]
        seqs, scores, finished = [[]], [0.0], []
        for step in range(max_new):
            n = len(seqs)
            if step == 0: es, ef = ps, pf
            else:
                t = wte(torch.tensor(seqs, device=dev))
                es = torch.cat([ps.expand(n, -1, -1), t], 1); ef = torch.cat([pf.expand(n, -1, -1), t], 1)
            lo = model.gpt(inputs_embeds=torch.cat([es, ef], 0)).logits[:, -1, :].float()
            ls, lf = torch.log_softmax(lo[:n], -1), torch.log_softmax(lo[n:], -1)
            lp = torch.log_softmax((1 + lam) * ls - lam * lf, -1)
            for k in range(n):
                for t_ in set(seqs[k]): lp[k, t_] = lp[k, t_] - log_rep
                for t_ in banned_ngram(seqs[k], no_repeat): lp[k, t_] = float('-inf')
            cand = []
            for k in range(n):
                tlp, tid = lp[k].topk(beams)
                cand += [(scores[k] + float(tlp[j]), k, int(tid[j])) for j in range(beams)]
            cand.sort(key=lambda x: x[0], reverse=True)
            new_seqs, new_scores = [], []
            for sc, k, tk in cand:
                if len(new_seqs) == beams: break
                if tk == eos_id:
                    seq = seqs[k] + [tk]; finished.append((sc / (len(seq) ** len_penalty), seq))
                else:
                    new_seqs.append(seqs[k] + [tk]); new_scores.append(sc)
            seqs, scores = new_seqs, new_scores
            if not seqs or len(finished) >= beams: break
        if not finished:   # unfinished beams only as fallback
            finished = [(scores[k] / (max(len(seqs[k]), 1) ** len_penalty), seqs[k]) for k in range(len(seqs))]
        results.append(max(finished, key=lambda x: x[0])[1] if finished else [])
    return results

def main(a):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok = build_tokenizer(); sid = add_style_tokens(tok)
    ck = torch.load(a.ckpt, map_location='cpu')
    model = StyleCaptioner(ck['clip_dim'], tok, ck['prefix_len']).to(dev)
    model.load_state_dict(ck['model']); model.eval()
    offset = ck.get('offset', None)
    styled = a.style != 'factual'
    mem = None
    if a.proj_mem and styled:
        mem = unit(torch.load(a.proj_mem, map_location='cpu')['emb'].float()).to(dev)
        print('projection memory', tuple(mem.shape), ' temp', a.proj_temp, ' mix', a.proj_mix, flush=True)
    print('offset_scale', a.offset_scale, ' beams', a.beams, ' guidance', a.guidance,
          ' top_p', a.top_p, ' top_k', a.top_k, ' temperature', a.temperature, ' seed', a.seed, flush=True)
    if (a.top_p > 0 or a.top_k > 0) and a.beams > 1: raise SystemExit('--top_p/--top_k need --beams 1 (sampling replaces beam search)')
    torch.manual_seed(a.seed)
    timg = torch.load(a.test_img, map_location='cpu')
    temb = {i: e.float() for i, e in zip(timg['ids'], timg['emb'])}
    order, refs = [], {}
    for ln in read_lines(a.test_cap):
        p = split_line(ln)
        if not p or not p[1]: continue
        img, cap = p
        if img in temb: refs.setdefault(img, []).append(cap)
        if img in temb and img not in order: order.append(img)
        if a.n_images and len(order) >= a.n_images: break
    out = {}
    for s in range(0, len(order), a.batch_size):
        ch = order[s:s+a.batch_size]
        raw = unit(torch.stack([temb[i] for i in ch])).to(dev)
        e = raw
        if offset is not None and styled: e = unit(e - a.offset_scale * offset.to(dev))
        if mem is not None: e = unit(a.proj_mix * project(raw, mem, a.proj_temp) + (1 - a.proj_mix) * e)
        code = torch.full((e.size(0),), sid[a.style], device=dev)
        if a.guidance > 0 and styled:
            fcode = torch.full((e.size(0),), sid['factual'], device=dev)
            outs = guided_decode(model, e, raw, code, fcode, a.guidance, max(a.beams, 1), a.max_new,
                                 tok.eos_token_id, 1.2, a.len_penalty, a.no_repeat)
        elif a.beams > 1:
            outs = model.generate_beam(e, code, max_new=a.max_new, eos_id=tok.eos_token_id,
                                       beams=a.beams, len_penalty=a.len_penalty,
                                       no_repeat=a.no_repeat)
        else:
            outs = model.generate(e, code, max_new=a.max_new, eos_id=tok.eos_token_id, no_repeat=a.no_repeat,
                                  top_p=a.top_p, temperature=a.temperature, top_k=a.top_k)
        for i, o in zip(ch, outs):
            out[i] = {'image_id': i, 'references': refs[i],
                      a.style: tok.decode(o, skip_special_tokens=True)}
        print(' ', min(s+a.batch_size, len(order)), '/', len(order), flush=True)
    json.dump(list(out.values()), open(a.out_json, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    for r in list(out.values())[:5]:
        print('\n', r['image_id']); print('  ref ', r['references'][0]); print('  gen ', r[a.style])
    print('wrote', a.out_json, len(out), 'images', flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--style', default='romantic', choices=['factual', 'romantic', 'humorous'])
    p.add_argument('--test_img', default='/kaggle/working/style_feats/test_images.pt')
    p.add_argument('--test_cap', default='/kaggle/working/splits/factual_test.txt')
    p.add_argument('--out_json', required=True)
    p.add_argument('--n_images', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--max_new', type=int, default=120)
    p.add_argument('--offset_scale', type=float, default=1.0)
    p.add_argument('--beams', type=int, default=1)
    p.add_argument('--len_penalty', type=float, default=1.0)
    p.add_argument('--no_repeat', type=int, default=3, help='block repeated n-grams, 0 = off')
    p.add_argument('--proj_mem', default=None, help='style text embeddings .pt (e.g. style_feats/humorous_train.pt)')
    p.add_argument('--proj_temp', type=float, default=0.01, help='projection softmax temperature (DeCap uses 0.01)')
    p.add_argument('--proj_mix', type=float, default=1.0, help='1 = pure projection, 0 = offset only')
    p.add_argument('--guidance', type=float, default=0.0, help='style guidance lambda, 0 = off')
    p.add_argument('--top_p', type=float, default=0.0, help='nucleus sampling (with --beams 1), 0 = off')
    p.add_argument('--top_k', type=int, default=0, help='top-k sampling (with --beams 1), 0 = off')
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=0)
    a, _ = p.parse_known_args(); main(a)
