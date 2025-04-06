import pandas as pd
import tempfile
import os
import copy
from typing import Dict, Tuple
import pytrec_eval
import argparse


def cal_mrr(qrels, results, k_values):
    mrr = {}
    for k in k_values:
        runs_topk = {query: dict(sorted(docs.items(), key=lambda x: x[1], reverse=True)[:k]) for query, docs in results.items()}
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'recip_rank'})
        scores = evaluator.evaluate(runs_topk)
        mrr[f"MRR@{k}"] = scores
    return mrr

def trec_eval(qrels: Dict[str, Dict[str, int]],
              results: Dict[str, Dict[str, float]],
              k_values: Tuple[int] = (10, 50, 100, 200, 1000)) -> Dict[str, float]:
    ndcg, _map, recall, mrr = {}, {}, {}, {}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        mrr[f"MRR@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string})
    scores = evaluator.evaluate(results)

    all_mrr = cal_mrr(qrels, results, k_values)

    for query_id in scores:
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            mrr[f"MRR@{k}"] += all_mrr[f"MRR@{k}"][query_id]["recip_rank"]

    def _normalize(m: dict) -> dict:
        return {k: round(v / len(scores), 5) for k, v in m.items()}

    ndcg = _normalize(ndcg)
    _map = _normalize(_map)
    recall = _normalize(recall)
    mrr = _normalize(mrr)

    all_metrics = {}
    for mt in [ndcg, _map, recall, mrr]:
        all_metrics.update(mt)

    return all_metrics


def remove_duplicate(response):
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
        else:
            print('duplicate')
    return new_response


def clean_response(response: str):
    new_response = ''
    for c in response:
        if not c.isdigit():
            new_response += ' '
        else:
            try:
                new_response += str(int(c))
            except:
                new_response += ' '
    new_response = new_response.strip()
    return new_response


class EvalFunction:
    @staticmethod
    def receive_responses(rank_results, responses, cut_start=0, cut_end=100):
        print('receive_responses', len(responses), len(rank_results))
        for i in range(len(responses)):
            response = responses[i]
            response = clean_response(response)
            response = [int(x) - 1 for x in response.split()]
            response = remove_duplicate(response)
            cut_range = copy.deepcopy(rank_results[i]['hits'][cut_start: cut_end])
            original_rank = [tt for tt in range(len(cut_range))]
            response = [ss for ss in response if ss in original_rank]
            response = response + [tt for tt in original_rank if tt not in response]
            for j, x in enumerate(response):
                rank_results[i]['hits'][j + cut_start] = {
                    'content': cut_range[x]['content'], 'qid': cut_range[x]['qid'], 'docid': cut_range[x]['docid'],
                    'rank': cut_range[j]['rank'], 'score': cut_range[j]['score']}
        return rank_results

    @staticmethod
    def trunc(qrels, run):
        qrels = get_qrels_file(qrels)
        # print(qrels)
        run = pd.read_csv(run, delim_whitespace=True, header=None)
        qrels = pd.read_csv(qrels, delim_whitespace=True, header=None)
        run[0] = run[0].astype(str)
        qrels[0] = qrels[0].astype(str)

        qrels = qrels[qrels[0].isin(run[0])]
        temp_file = tempfile.NamedTemporaryFile(delete=False).name
        qrels.to_csv(temp_file, sep='\t', header=None, index=None)
        return temp_file

    @staticmethod
    def main(args_qrel, args_run):

        # args_qrel = EvalFunction.trunc(args_qrel, args_run)

        assert os.path.exists(args_qrel)
        assert os.path.exists(args_run)

        with open(args_qrel, 'r') as f_qrel:
            qrel = pytrec_eval.parse_qrel(f_qrel)

        with open(args_run, 'r') as f_run:
            run = pytrec_eval.parse_run(f_run)

        all_metrics = trec_eval(qrel, run, k_values=(1, 5, 10, 50, 100, 1000))
        print(all_metrics)
        return all_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--qrel_file')
    parser.add_argument('--run_file')
    args = parser.parse_args()
    EvalFunction.main(args.qrel_file, args.run_file)
