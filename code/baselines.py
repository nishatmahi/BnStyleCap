# baselines.py
# Zero-shot MLLM baselines on the same test images, same output format as generate.py, so
# evaluate.py scores them identically. Also records efficiency (params, peak GPU memory, seconds/caption).
#
#   python baselines.py --model qwen  --style humorous --out_json /kaggle/working/qwen_humorous.json
#   python baselines.py --model llama --style romantic --out_json /kaggle/working/llama_romantic.json
#   python baselines.py --model ours  --ckpt sc_paired/paired_humorous_romantic_best.pth --style humorous \
#                       --out_json /kaggle/working/eff_ours.json --n_images 100 --device cpu   (efficiency only)
#
# LLaMA-3.2-Vision is gated: accept the licence on its Hugging Face page and add a Kaggle secret HF_TOKEN.
# Kaggle T4 x2: Qwen fits in fp16 across both GPUs; LLaMA 11B uses 4-bit (--load_4bit) if memory is short.
# Gemma 3 12B: 4-bit weights with float32 compute (Gemma overflows in fp16 and T4 has no bf16).
import os, re, json, time, argparse
import torch
from PIL import Image

MODELS = {'qwen': 'Qwen/Qwen2.5-VL-7B-Instruct', 'llama': 'meta-llama/Llama-3.2-11B-Vision-Instruct',
          'gemma': 'google/gemma-3-12b-it'}   # gemma: Kaggle model path via --model_path
PROMPTS = {
    'factual':  'Describe this image in one short, plain, factual sentence in Bangla (Bengali script). '
                'Reply with the sentence only.',
    'humorous': 'Write one short humorous caption for this image in Bangla (Bengali script). '
                'It must describe what is in the image and be funny. Reply with the caption only.',
    'romantic': 'Write one short romantic caption for this image in Bangla (Bengali script). '
                'It must describe what is in the image with warmth, love or tenderness. Reply with the caption only.',
}

def strip_ext(n):
    for e in ('.jpg', '.jpeg', '.png'):
        if n.lower().endswith(e): return n[:-len(e)]
    return n

def test_ids(test_cap, n):
    order = []
    for l in open(test_cap, encoding='utf-8'):
        p = [x.strip() for x in re.split(r'#\d*', l.strip()) if x.strip()]
        if len(p) >= 2:
            i = strip_ext(p[0])
            if i not in order: order.append(i)
        if n and len(order) >= n: break
    return order

def img_path(d, i):
    for e in ('.jpg', '.jpeg', '.png'):
        if os.path.exists(os.path.join(d, i + e)): return os.path.join(d, i + e)

def clean(t):
    t = t.strip().split('\n')[0].strip().strip('"“”\'')
    return t

def load_mllm(name, load_4bit, path=None):
    from transformers import AutoProcessor
    dt = torch.float32 if name == 'gemma' else torch.float16
    kw = {'torch_dtype': dt, 'device_map': 'auto'}
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kw['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dt)
    token = os.environ.get('HF_TOKEN')
    if name == 'qwen':
        from transformers import Qwen2_5_VLForConditionalGeneration as M
    elif name == 'gemma':
        from transformers import Gemma3ForConditionalGeneration as M
    else:
        from transformers import MllamaForConditionalGeneration as M
    src = path or MODELS[name]
    model = M.from_pretrained(src, token=token, **kw).eval()
    proc = AutoProcessor.from_pretrained(src, token=token)
    return model, proc

@torch.no_grad()
def caption_mllm(model, proc, name, img, prompt, max_new):
    if name == 'gemma':
        msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': img}, {'type': 'text', 'text': prompt}]}]
        inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                       return_tensors='pt').to(model.device)
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
        return clean(proc.decode(out[0, inp['input_ids'].shape[1]:], skip_special_tokens=True))
    msgs = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True)
    if name == 'qwen': inp = proc(text=[text], images=[img], return_tensors='pt')
    else: inp = proc(img, text, add_special_tokens=False, return_tensors='pt')
    inp = inp.to(model.device)
    out = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
    return clean(proc.decode(out[0, inp['input_ids'].shape[1]:], skip_special_tokens=True))

def main(a):
    ids = test_ids(a.test_cap, a.n_images)
    out, times = [], []
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    if a.model == 'ours':
        # efficiency of our model on the same images (captions come from generate.py)
        from open_clip import create_model_from_pretrained
        from stage1 import StyleCaptioner, build_tokenizer, add_style_tokens, unit
        dev = torch.device(a.device)
        clip, pre = create_model_from_pretrained({'base': 'nllb-clip-base-siglip', 'large': 'nllb-clip-large-siglip'}[a.clip], 'v1', device=dev); clip.eval()
        tok = build_tokenizer(); sid = add_style_tokens(tok)
        ck = torch.load(a.ckpt, map_location='cpu')
        model = StyleCaptioner(ck['clip_dim'], tok, ck['prefix_len']).to(dev)
        model.load_state_dict(ck['model']); model.eval(); off = ck.get('offset')
        n_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in clip.visual.parameters())
        for i in ids:
            t0 = time.perf_counter()
            with torch.no_grad():
                e = unit(clip.encode_image(pre(Image.open(img_path(a.img_dir, i)).convert('RGB')).unsqueeze(0).to(dev)).float())
                if off is not None and a.style != 'factual': e = unit(e - a.offset_scale * off.to(dev))
                code = torch.full((1,), sid[a.style], device=dev)
                o = model.generate_beam(e, code, eos_id=tok.eos_token_id, beams=a.beams) if a.beams > 1 \
                    else model.generate(e, code, eos_id=tok.eos_token_id)
            times.append(time.perf_counter() - t0)
            out.append({'image_id': i, a.style: tok.decode(o[0], skip_special_tokens=True)})
    else:
        model, proc = load_mllm(a.model, a.load_4bit, a.model_path)
        n_params = sum(p.numel() for p in model.parameters())
        for k, i in enumerate(ids):
            img = Image.open(img_path(a.img_dir, i)).convert('RGB')
            t0 = time.perf_counter()
            cap = caption_mllm(model, proc, a.model, img, PROMPTS[a.style], a.max_new)
            times.append(time.perf_counter() - t0)
            out.append({'image_id': i, a.style: cap})
            if k % 50 == 0: print(f'  {k}/{len(ids)}  {i}: {cap}', flush=True)
    json.dump(out, open(a.out_json, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    eff = {'system': a.model, 'style': a.style, 'params_M': round(n_params / 1e6, 1),
           'peak_gpu_GB': round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0.0,
           'sec_per_caption': round(sum(times) / max(len(times), 1), 4), 'device': a.device if a.model == 'ours' else 'cuda',
           'n': len(times)}
    with open(a.eff_out, 'a', encoding='utf-8') as f: f.write(json.dumps(eff) + '\n')
    print('wrote', a.out_json, len(out), 'captions   efficiency', eff, flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, choices=['qwen', 'llama', 'gemma', 'ours'])
    p.add_argument('--model_path', default=None, help='local model directory (e.g. Kaggle Gemma 3)')
    p.add_argument('--style', default='humorous', choices=list(PROMPTS))
    p.add_argument('--out_json', required=True)
    p.add_argument('--ckpt', default=None)
    p.add_argument('--test_cap', default='/kaggle/working/splits/factual_test.txt')
    p.add_argument('--img_dir', default='/kaggle/input/datasets/kaggleperfect/dataset/data/Images')
    p.add_argument('--n_images', type=int, default=1000)
    p.add_argument('--max_new', type=int, default=60)
    p.add_argument('--beams', type=int, default=1)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--load_4bit', action='store_true')
    p.add_argument('--clip', default='large', choices=['base', 'large'], help='our model: CLIP used for its features')
    p.add_argument('--offset_scale', type=float, default=1.0, help='our model: same offset as generation')
    p.add_argument('--eff_out', default='/kaggle/working/efficiency.jsonl')
    a, _ = p.parse_known_args(); main(a)
