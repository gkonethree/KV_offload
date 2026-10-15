import json
import os, sys
O = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
names=['pq_r','b32_f50_r','l32_f50','b128_f50','l128_f50','b32_f99','l32_f99','b128_f99','l128_f99','long_pq','long_b128','long_l128']
ref={'s':json.load(open(f'{O}/pq.json'))['tokens'],'l':json.load(open(f'{O}/long_pq.json'))['tokens']}
PB=16384
for n in names:
    d=json.load(open(f'{O}/{n}.json')); r=d['runs'][-1]; c=r['counters']; lp=r.get('layer_pool') or {}; m=r['mem']; st=max(m['steps'],1)
    cfg=d['config']
    cp = c['block_blocks']*22*PB if cfg=='block' else lp.get('bytes_h2d',0)
    layer_blocks = c['block_blocks']*22 if cfg=='block' else lp.get('blocks_fetched',0)
    hits,miss=lp.get('sel_hits',0),lp.get('sel_misses',0)
    T=r['timing']; g=lambda k:T.get(k,{}).get('total_ms',0)/1e3
    fetch=g('wrap.fetch'); book=g('wrap.discover')+g('wrap.pl_mask'); pqt=g('pq.append')+g('pq.select')+g('pq.luts')+g('pq.adc'); dec=g('wrap.decode')
    print(n, cfg, d['slots'] if cfg!='pq' else '-', d['fraction'] if cfg!='pq' else '-', d['max_model_len'], d['prompt_tokens'], d['new_tokens'], d['prompt_tokens']+d['new_tokens'],
          round(d['tpot_ms'],1), [round(x,1) for x in d['tpot_all']], round(r['time_s'],1), round(d['warmup_s']),
          d['tokens']==ref['l' if n.startswith('long') else 's'], d['tokens_stable'],
          'cp_MB', round(cp/1e6,1), 'cp_MB_per_tok', round(cp/1e6/d['new_tokens'],2), 'layerblocks', layer_blocks, 'fetchcalls', c['block_fetch_calls'] if cfg=='block' else lp.get('fetch_calls',0),
          'chunks', c['chunk_merges'], 'evict', lp.get('evictions','-'), 'skip', lp.get('skipped','-'),
          'hit', f"{100*hits/(hits+miss):.1f}" if hits+miss else '-',
          'gpuavg', round(m['sum_gpu']/st,1), 'peaktot', m['peak_total'],
          'fetch_s', round(fetch,1), 'book_s', round(book,1), 'pq_s', round(pqt,1), 'dec_s', round(dec,2), 'patch_s', round(g('wrap.patch'),1), 'off_s', round(g('sched.offload'),2), 'fit_s', round(g('pq.fit'),1))
