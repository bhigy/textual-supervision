import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

from platalea.utils.get_best_score import get_best_score, get_best_full

plt.style.use('seaborn-darkgrid')
font = {'size': 12}
matplotlib.rc('font', **font)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


def extract_matched_results(root, pattern, path2type_fn):
    comp_ptrn = re.compile(pattern)
    res = {}
    for d in Path(root).iterdir():
        matched = comp_ptrn.match(str(d.name))
        if matched is not None:
            tmp_res = res
            for m in matched.groups()[:-1]:
                tmp_res = tmp_res.setdefault(m, {})
            score = get_best_score(d / 'result.json', path2type_fn(d))
            tmp_res[matched.groups()[-1]] = score
    return res


def extract_matched_metric(root, pattern, path2type_fn, metric):
    comp_ptrn = re.compile(pattern)
    res = {}
    for d in Path(root).iterdir():
        etype = path2type_fn(d)
        metric_accessor_fn = get_metric_accessor(etype, metric)
        matched = comp_ptrn.match(str(d.name))
        if matched is not None:
            tmp_res = res
            for m in matched.groups()[:-1]:
                tmp_res = tmp_res.setdefault(m, {})
            score = get_best_full(d / 'result.json', etype)
            tmp_res[matched.groups()[-1]] = metric_accessor_fn(score)
    return res


def extract_results(root, expnames, ds_factors, replids, name2type_fn,
                    tag=''):
    res = np.zeros([len(expnames), len(ds_factors), len(replids)])
    for i, ename in enumerate(expnames):
        etype = name2type_fn(ename)
        for j, ds in enumerate(ds_factors):
            for k, rid in enumerate(replids):
                pattern = '{}{}-ds{}-{}-*'.format(ename, tag, ds, rid)
                paths = sorted(Path(root).glob(pattern))
                if len(paths) != 1:
                    msg = 'Pattern {} matches {} folders'
                    raise ValueError(msg.format(pattern, len(paths)))
                score = get_best_score(paths[0] / 'result.json', etype)
                res[i, j, k] = score
    return res


def extract_metric(root, expnames, ds_factors, replids, name2type_fn,
                   tag='', metric='R@10'):
    res = np.zeros([len(expnames), len(ds_factors), len(replids)])
    for i, ename in enumerate(expnames):
        etype = name2type_fn(ename)
        metric_accessor_fn = get_metric_accessor(etype, metric)
        for j, ds in enumerate(ds_factors):
            for k, rid in enumerate(replids):
                pattern = '{}{}-ds{}-{}-*'.format(ename, tag, ds, rid)
                paths = sorted(Path(root).glob(pattern))
                if len(paths) != 1:
                    msg = 'Pattern {} matches {} folders'
                    raise ValueError(msg.format(pattern, len(paths)))
                scores = get_best_full(paths[0] / 'result.json', etype)
                res[i, j, k] = metric_accessor_fn(scores)
    return res


def name2type(expname):
    expname = expname.split('-')[0]
    exptype = {'asr': 'asr', 'basic': 'retrieval', 'mtl': 'mtl',
               'pip': 'retrieval', 'text': 'retrieval'}
    return exptype[expname]


def path2type(path):
    expname = Path(path).name.split('-')[0]
    return name2type(expname)


def name2type_jp(expname):
    expname = expname.split('-')[0]
    exptype = {'asr': 'slt', 'basic': 'retrieval', 'mtl': 'mtl',
               'pip': 'retrieval', 'text': 'retrieval'}
    return exptype[expname]


def path2type_jp(path):
    expname = Path(path).name.split('-')[0]
    return name2type_jp(expname)


def dict2np(res):
    if type(res) == dict:
        return [dict2np(v) for v in res.values()]
    else:
        return res


def plot_downsampling(fname='downsampling_text.pdf'):
    # Extracting results
    basic_default_results = extract_matched_results(
        'runs', '(basic-default)-([abc])-.*', path2type)
    basic_default_score = np.mean(dict2np(basic_default_results))
    expnames = ['asr', 'text-image', 'pip-ind', 'pip-seq', 'mtl-asr',
                 'mtl-st']
    exp_legend = ['asr', 'text-image', 'pipe-ind', 'pipe-seq',
                  'mtl-transcribe', 'mtl-match']
    ds_factors = [1, 3, 9, 27, 81, 243]
    ds_factors_text = [str(i).zfill(3) for i in ds_factors]
    replids = ['a', 'b', 'c']
    res = extract_results('runs', expnames, ds_factors_text, replids,
                          name2type)
    res = np.mean(res, axis=2)

    # Plotting
    xticklabels = ['34 h', '11.3 h', '3.8 h', '1.3 h', '25 mins', '8 mins']
    fig, ax = plt.subplots()
    ax.set_xlabel('Amount of transcribed data available for training (total speech duration)')
    plt.xticks(range(len(xticklabels)), xticklabels, size='small')
    ax.set_ylabel('R@10')
    ax.plot([basic_default_score] * len(ds_factors), 'r--',
            label='speech-image')
    for i in range(1, len(expnames)):
        ax.plot(res[i, :], '.-', label=exp_legend[i])
    ax.legend()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(fname)


def plot_downsampling_jp(fname='downsampling_text_jp.pdf'):
    # Extracting results
    basic_default_results = extract_matched_results(
        'runs', '(basic-default)-jp-([abc])-.*', path2type)
    basic_default_score = np.mean(dict2np(basic_default_results))
    expnames = ['asr', 'text-image', 'pip-ind', 'pip-seq', 'mtl-asr',
                 'mtl-st']
    exp_legend = ['slt', 'text-image', 'pipe-ind', 'pipe-seq',
                  'mtl-translate', 'mtl-match']
    ds_factors = [1, 3, 9, 27, 81, 243]
    ds_factors_text = [str(i).zfill(3) for i in ds_factors]
    replids = ['a', 'b', 'c']
    res = extract_results('runs', expnames, ds_factors_text, replids,
                          name2type_jp, '-jp')
    res = np.mean(res, axis=2)

    # Plotting
    fig, ax = plt.subplots()
    xticklabels = ['13.6 h', '4.5 h', '1.5 h', '30 mins', '10 mins', '3 mins']
    ax.set_xlabel('Amount of translated data available for training (total speech duration)')
    plt.xticks(range(len(xticklabels)), xticklabels, size='small')
    ax.set_ylabel('R@10')
    ax.plot([basic_default_score] * len(ds_factors), 'r--',
            label='speech-image')
    for i in range(1, len(expnames)):
        ax.plot(res[i, :], '.-', label=exp_legend[i])
    ax.legend()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(fname)


def extract_results_beam_decoding(root, exp_prefix, exptype):
    ds_factors = [1, 3, 9, 27, 81, 243]
    replids = ['a', 'b', 'c']
    res = np.zeros([len(ds_factors), len(replids)])
    for i, df in enumerate(ds_factors):
        df_text = str(df).zfill(3)
        for j, rid in enumerate(replids):
            pattern = '{}-ds{}-{}-*'.format(exp_prefix, df_text, rid)
            paths = sorted(Path(root).glob(pattern))
            if len(paths) != 1:
                msg = 'Pattern {} matches {} folders'
                raise ValueError(msg.format(pattern, len(paths)))
            score = get_best_score(paths[0] / 'result_beam.json', exptype)
            res[i, j] = score
    res = np.mean(res, axis=1)
    return res


def get_metric_accessor(experiment_type, metric):
    ''' Returns an accessor function that accesses the required metric ('R@1',
    'R@5', 'R@10' or 'Medr') given a record of all metrics for the specified
    experiment type ('retrival' or 'mtl') '''
    if metric in ['R@1', 'R@5', 'R@10']:
        n = metric[2:]
        if experiment_type == 'retrieval':
            return lambda x, n=n: x['recall'][n]
        elif experiment_type == 'mtl':
            return lambda x, n=n: x['SI']['recall'][n]
    else:
        if experiment_type == 'retrieval':
            return lambda x: x['medr']
        elif experiment_type == 'mtl':
            return lambda x: x['SI']['medr']


def extract_results_test_set(root, tag='', metric='R@10'):
    if tag != '':
        tag = '-' + tag
    exp_names = ['pip-seq', 'mtl-asr']
    exp_types = ['retrieval', 'mtl']
    replids = ['a', 'b', 'c']
    res = np.zeros([len(exp_names), len(replids)])
    for i, exp in enumerate(exp_names):
        metric_accessor_fn = get_metric_accessor(exp_types[i], metric)
        for j, rid in enumerate(replids):
            pattern = '{}{}-ds001-{}-*'.format(exp, tag, rid)
            paths = sorted(Path(root).glob(pattern))
            if len(paths) != 1:
                msg = 'Pattern {} matches {} folders'
                raise ValueError(msg.format(pattern, len(paths)))
            scores = get_best_full(paths[0] / 'result_test.json', exp_types[i])
            res[i, j] = metric_accessor_fn(scores)
    res = np.mean(res, axis=1)
    return res


def extract_results_comparison(root):
    experiments = {'basic-default': 'retrieval', 'text-image': 'retrieval',
                   'pip-ind': 'retrieval', 'pip-seq': 'retrieval',
                   'mtl-asr': 'mtl', 'mtl-st': 'mtl'}
    replids = ['a', 'b', 'c']
    res = np.zeros([len(experiments.keys()), len(replids)])
    for i, exp in enumerate(experiments.items()):
        for j, rid in enumerate(replids):
            pattern = '{}-comp-{}-*'.format(exp[0], rid)
            paths = sorted(Path(root).glob(pattern))
            if len(paths) != 1:
                msg = 'Pattern {} matches {} folders'
                raise ValueError(msg.format(pattern, len(paths)))
            score = get_best_score(paths[0] / 'result.json', exp[1])
            res[i, j] = score
    res = np.mean(res, axis=1)
    return res


def extract_metric_baselines(metric='R@10'):
    si_results_en = extract_matched_metric(
        'runs', '(basic-default)-([abc])-.*', path2type, metric)
    si_score_en = np.mean(dict2np(si_results_en))
    si_results_en_reduced = extract_matched_metric(
        'runs', '(basic-default)-comp-([abc])-.*', path2type, metric)
    si_score_en_reduced = np.mean(dict2np(si_results_en_reduced))
    si_results_jp = extract_matched_metric(
        'runs', '(basic-default)-jp-([abc])-.*', path2type_jp, metric)
    si_score_jp = np.mean(dict2np(si_results_jp))
    ti_results_en = extract_matched_metric(
        'runs', '(text-image)-ds001-([abc])-.*', path2type, metric)
    ti_score_en = np.mean(dict2np(ti_results_en))
    ti_results_en_reduced = extract_matched_metric(
        'runs', '(text-image)-comp-([abc])-.*', path2type, metric)
    ti_score_en_reduced = np.mean(dict2np(ti_results_en_reduced))
    ti_results_jp = extract_matched_metric(
        'runs', '(text-image)-jp-ds001-([abc])-.*', path2type_jp, metric)
    ti_score_jp = np.mean(dict2np(ti_results_jp))
    return np.array([[si_score_en, si_score_en_reduced, si_score_jp],
                     [ti_score_en, ti_score_en_reduced, ti_score_jp]])


def plot_figure_3():
    plot_downsampling('figure_3_left.pdf')
    plot_downsampling_jp('figure_3_right.pdf')


def print_table_1():
    scores = extract_metric_baselines()
    print(scores)


def print_table_2():
    expnames = ['pip-ind', 'pip-seq', 'mtl-asr', 'mtl-st']
    ds_factors = ['001']
    replids = ['a', 'b', 'c']
    res_en = extract_results('runs', expnames, ds_factors, replids,
                             name2type)
    scores_en = np.mean(res_en, axis=2)
    res_jp = extract_results('runs', expnames, ds_factors, replids,
                             name2type_jp, '-jp')
    scores_jp = np.mean(res_jp, axis=2)
    scores_en_reduced = []
    for ename in expnames:
        res = extract_matched_results(
            'runs', '({})-comp-([abc])-.*'.format(ename), path2type)
        scores_en_reduced.append(np.mean(dict2np(res)))
    scores_en_reduced = np.array([scores_en_reduced]).T
    scores = np.concatenate([scores_en, scores_en_reduced, scores_jp], axis=1)
    print(scores)


def print_table_3():
    scores_en = extract_results_test_set('runs', '')
    scores_jp = extract_results_test_set('runs', 'jp')
    scores = np.array([scores_en, scores_jp]).T
    print(scores)


def print_table_6():
    scores_r1 = extract_metric_baselines('R@1')
    scores_r5 = extract_metric_baselines('R@5')
    scores_medr = extract_metric_baselines('Medr')
    scores = np.concatenate([scores_r1, scores_r5, scores_medr])
    scores = scores.T.reshape([9, 2]).T
    print(scores)


def print_table_7():
    expnames = ['pip-ind', 'pip-seq', 'mtl-asr', 'mtl-st']
    ds_factors = ['001']
    replids = ['a', 'b', 'c']
    scores_en = np.zeros([4, 3])
    scores_jp = np.zeros([4, 3])
    scores_en_reduced = np.zeros([4, 3])
    for i, metric in enumerate(['R@1', 'R@5', 'Medr']):
        res = extract_metric('runs', expnames, ds_factors, replids,
                                 path2type, '', metric)
        scores_en[:, i] = np.mean(res, axis=2).squeeze()
        res = extract_metric('runs', expnames, ds_factors, replids,
                                 path2type_jp, '-jp', metric)
        scores_jp[:, i] = np.mean(res, axis=2).squeeze()
        for j, ename in enumerate(expnames):
            res = extract_matched_metric(
                'runs', '({})-comp-([abc])-.*'.format(ename), path2type,
                metric)
            scores_en_reduced[j, i] = np.mean(dict2np(res))
    scores = np.concatenate([scores_en, scores_en_reduced, scores_jp], axis=1)
    print(scores)
    pass


def print_table_8():
    scores_en = np.zeros([2, 3])
    scores_jp = np.zeros([2, 3])
    for i, metric in enumerate(['R@1', 'R@5', 'Medr']):
        scores_en[:, i] = extract_results_test_set('runs', '', metric)
        scores_jp[:, i] = extract_results_test_set('runs', 'jp', metric)
    scores = np.concatenate([scores_en, scores_jp], axis=1)
    print(scores)


def print_table_9():
    print(extract_results_beam_decoding('runs', 'asr', 'asr'))


def print_table_10():
    print(extract_results_beam_decoding('runs', 'asr-jp', 'slt'))
