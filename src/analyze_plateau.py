"""Analyze L15 saturation and bound mixed reward-field effects from saved data.

No model replay or new training. The archive retains gradient norms and cosines
for pairs (5,45) and (15,45), but not (5,15). Positive-semidefinite Gram geometry
bounds the missing inner product, hence the local effect of the equally mixed
reward field on the measured L15 prompt set. This excludes entropy, stochastic
sampling and Adam; it does not reconstruct the actual training update.
"""
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np


def summarize(archive, output):
    archive = Path(archive)
    with zipfile.ZipFile(archive) as z:
        def read(name):
            return list(csv.DictReader(io.StringIO(z.read(name).decode())))
        gradient = read('diagnostics/gradient_seed101/measurements.csv')
        transfer = read('diagnostics/transfer_seed101/measurements.csv')
        evaluation = read('runs/mixed5_15_45_seed101/eval.csv')
        training = read('runs/mixed5_15_45_seed101/train.csv')
    g = {(int(r['step']),int(r['repeat']),int(r['length'])):r for r in gradient if r['kind']=='gradient'}
    t = {(int(r['step']),int(r['repeat']),int(r['source'])):r for r in transfer
         if r['kind']=='transfer' and float(r['epsilon'])==.01}
    bounds = []
    for step in range(0,40001,2000):
        for repeat in range(3):
            n5,n15,n45=[float(g[(step,repeat,l)]['reward_gradient_norm']) for l in (5,15,45)]
            c5=float(t[(step,repeat,5)]['gradient_cosine'])
            c15=float(t[(step,repeat,15)]['gradient_cosine'])
            c5,c15=np.clip(c5,-1,1),np.clip(c15,-1,1)
            radius=np.sqrt(max(0.,(1-c5*c5)*(1-c15*c15)))
            cmin,cmax=c5*c15-radius,c5*c15+radius
            # J15 = mean P; g_l = grad J_l / l.
            # grad J15 . (g5+g15+g45)/3 = 5 * (g15.g5 + ||g15||^2 + g15.g45).
            cross45=n15*n45*c15
            lower=5*(n15*n5*cmin+n15*n15+cross45)
            upper=5*(n15*n5*cmax+n15*n15+cross45)
            sign='positive' if lower>0 else ('negative' if upper<0 else 'undetermined')
            bounds.append({'step':step,'repeat':repeat,'norm_L5':n5,'norm_L15':n15,'norm_L45':n45,
                           'cosine_L5_L45':float(c5),'cosine_L15_L45':float(c15),
                           'cosine_L5_L15_lower':float(cmin),'cosine_L5_L15_upper':float(cmax),
                           'L15_self_derivative_in_mixture':5*n15*n15,
                           'L45_cross_derivative_in_mixture':5*cross45,
                           'L15_mixed_field_derivative_lower':float(lower),
                           'L15_mixed_field_derivative_upper':float(upper),'bounded_sign':sign})
    # Check the bound algebra against random vectors, with a deterministic CPU seed.
    rng=np.random.default_rng(73115)
    for _ in range(1000):
        a,b,c=rng.normal(size=(3,7))
        a,b,c=[v/np.linalg.norm(v) for v in (a,b,c)]
        ac,bc=a@c,b@c
        radius=np.sqrt(max(0.,(1-ac*ac)*(1-bc*bc)))
        assert ac*bc-radius-1e-12 <= a@b <= ac*bc+radius+1e-12
    steps=list(range(0,40001,100))
    curves={l:np.array([np.mean([float(g[(s,r,l)]['reward_gradient_norm']) for r in range(3)]) for s in steps])
            for l in (5,15,45)}
    final_p=np.concatenate([json.loads(g[(40000,r,15)]['per_prompt_probabilities']) for r in range(3)])
    high=final_p>.5
    probability_groups={}
    for step in (10300,10500,12000,20000,30000,40000):
        p=np.concatenate([json.loads(g[(step,r,15)]['per_prompt_probabilities']) for r in range(3)])
        probability_groups[str(step)]={label:{'n':int(mask.sum()),'mean':float(p[mask].mean()),
                                             'min':float(p[mask].min()),'max':float(p[mask].max())}
                                        for label,mask in [('eventually_high',high),('eventually_low',~high)]}
    stable_metrics={}
    for length in (5,15,45):
        ev=sorted([r for r in evaluation if int(r['length'])==length],key=lambda r:int(r['step']))
        result={}
        for metric in ('attention_hit','greedy_trajectory_correct'):
            target=float(ev[-1][metric])
            last_different=max((i for i,r in enumerate(ev) if float(r[metric])!=target),default=-1)
            result[metric+'_constant_from']=int(ev[last_different+1]['step'])
            result[metric+'_final']=target
        stable_metrics[str(length)]=result
    late=[r for r in bounds if r['step']>=12000]
    late_train=[r for r in training if int(r['step'])>=12000]
    evidence={
        'source':'src/analyze_plateau.py','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'gradient_summary':{str(l):{'peak_step':steps[int(curves[l].argmax())],
                                  'peak':float(curves[l].max()),'final':float(curves[l][-1]),
                                  'peak_over_final':float(curves[l].max()/curves[l][-1])} for l in curves},
        'probability_groups':probability_groups,'fixed_evaluation_counts':stable_metrics,
        'observed_L15_states':{str(s):{k:float(next(r for r in evaluation if int(r['length'])==15 and int(r['step'])==s)[k])
                                     for k in ('greedy_success','sampled_success','attention_hit','attention_mass','correct_transition_prob','entropy','greedy_trajectory_correct')}
                               for s in (10500,12000,20000,40000)},
        'actual_update_norm_late':{'mean':float(np.mean([float(r['parameter_update_norm']) for r in late_train])),
                                   'minimum':min(float(r['parameter_update_norm']) for r in late_train)},
        'final_L15_output_confidence':{
            'mean_entropy_nats':float(next(r for r in evaluation if int(r['length'])==15 and int(r['step'])==40000)['entropy']),
            'mean_max_probability_lower_bound':float(np.exp(-float(next(r for r in evaluation if int(r['length'])==15 and int(r['step'])==40000)['entropy']))),
            'reason':'H(p) >= -log(max p), hence E[max p] >= exp(-E[H]); averaging follows the recorded rollout states and prompts.'},
        'mixed_reward_field':{'definition':'G = (g5+g15+g45)/3, gL = gradient_Q mean(P_L)/L, three independently fixed 16-prompt sets per length',
                              'bound':'dJ15[G] = 5*(||g15||^2 + n15*n45*c15,45 + n15*n5*c5,15); c5,15 in c5,45*c15,45 +/- sqrt((1-c5,45^2)*(1-c15,45^2))',
                              'late_from':12000,'n_late':len(late),
                              'late_sign_counts':{sign:sum(r['bounded_sign']==sign for r in late) for sign in ('positive','negative','undetermined')},
                              'all_bounds':bounds,
                              'scope':'Local expected reward field on retained prompt sets; excludes entropy, minibatch noise and Adam. Not the observed training direction.'},
        'numerical_check':'Gram bounds verified on 1000 seeded random unit-vector triples',
        'interpretation':'Persistent polarized prompt outcomes and fixed aggregate attention-hit/full-path counts coexist with collapsing L15 reward gradients and nonzero overall parameter updates. This supports saturation of an incomplete policy; it does not identify the implementation choice that formed that policy.',
    }
    Path(output).write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    return evidence
