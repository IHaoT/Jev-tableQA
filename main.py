
from configs import GEMINI_KEYS,hyperparameter
from GraphRetriever.graph_retriver import GraphRetriever

from Generator.openai_api import ChatGPTTool
from GraphRetriever.dense_retriever import load_dense_retriever
from Generator.Gemini_model import GeminiTool
# from vertexai.preview import tokenization
import argparse

import tiktoken
from dashscope import get_tokenizer
from iterative_reasoning import GraphReasoner
from compute_score import eval_ex_match,LLM_eval
import json
import jsonlines

global tokenizer
logger = None

def augments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default='')
    parser.add_argument("--base_url", type=str, default='')
    parser.add_argument("--key", type=str,default='key')
    parser.add_argument("--dataset", type=str, default='')
    parser.add_argument("--qa_path", type=str, required=True)
    parser.add_argument("--table_folder", type=str, )

    parser.add_argument("--max_iteration_depth", type=int, required=True)
    # 官方脚本 start.sh 未传 --start/--end，这里改成可选：默认跑全量。
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--embed_cache_dir", type=str)
    parser.add_argument("--embedder_path", type=str,default='')

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output_path", type=str, default='')
    # 只跑指定下标的子集（JSON 文件，内容形如 [0,1,450,...]）。给定时忽略 --start/--end。
    parser.add_argument("--index_file", type=str, default='')
    args = parser.parse_args()

    return args

def load_model(args):
    if 'gemini' in args.model:
        gemini_key_index = 0
        gemini_key = GEMINI_KEYS[gemini_key_index]
        model = GeminiTool(gemini_key, args)
    else:
        model = ChatGPTTool(args)

    dense_retriever = load_dense_retriever(args)

    return  model,dense_retriever

def _select(items, args):
    """按 --index_file 选子集，否则退回 --start/--end 切片。"""
    if args.index_file:
        with open(args.index_file, 'r', encoding='utf-8') as f:
            idx = json.load(f)
        return [items[i] for i in idx]
    return items[args.start:args.end]


def load_data(args):

    querys,answers,table_captions,tables,table_paths = [],[],[],[],[]

    if args.dataset in ('hitab','Hitab'):
        qas = []
        with open(args.qa_path, "r+", encoding='utf-8') as f:
            for item in jsonlines.Reader(f):
                qas.append(item)
        qas = _select(qas, args)

        for qa in qas:
            table_path = args.table_folder + qa['table_id'] + '.json'
            with open(table_path, "r+", encoding='utf-8') as f:
                table = json.load(f)
            table_captions.append(table['title'])
            answers.append('|'.join([str(i) for i in qa['answer']]))
            querys.append( qa['question'])
            table_paths.append(table_path)
            tables.append(table['texts'])
    elif args.dataset in ('AIT-QA','ait-qa'):
        with open(args.qa_path, 'r', encoding='utf-8') as f:
            qas = json.load(f)
        qas = _select(qas, args)

        for qa in qas:
            tables.append(qa['table'])
            answers.append('|'.join([str(i) for i in qa['answers']]))
            querys.append( qa['question'])
            table_captions.append('')
            table_paths.append(qa)
    return querys,answers,table_captions,tables,table_paths


def run_question(query, answer, caption, table, table_path, args, model, dense_retriever):
    """跑单题；单题失败重试 3 次，仍失败返回 None，不中断整轮。"""
    error = 3
    while error > 0:
        try:
            # 重试时重建，避免上一轮已改写的检索器/推理状态污染这一轮。
            graph_retriever = GraphRetriever(table_path, model, dense_retriever, args.dataset, table_cation=caption)
            graph_reasoner = GraphReasoner(args, model, query, table, caption, graph_retriever, args.dataset)
            return graph_reasoner.iterative_reasoning()
        except UserWarning as v:
            print(query + '\t' + args.dataset + '\t' + '不满足安全协议' + v.__str__())
            return None
        except Exception as e:
            print(query + '\t' + '报错了' + '\t' + e.__str__())
            error -= 1
    return None


def main():
    args = augments()

    model, dense_retriever = load_model(args)
    querys, answers, table_captions, tables,table_paths = load_data(args)

    total_num,EM,LLM_EVAL = 0,0,0
    records = []
    # 结果里的 index 用原始数据集下标，便于分段跑完后按题号合并。
    base = 0 if args.index_file else args.start
    for index, (query, answer, caption, table, table_path) in enumerate(
            zip(querys, answers, table_captions, tables, table_paths)):
        global_index = index + base
        output = run_question(query, answer, caption, table, table_path, args, model, dense_retriever)
        if output is None:
            records.append({'index': global_index, 'question': query, 'gold': answer,
                            'pred': None, 'em': None, 'status': 'failed'})
            continue

        print('模型回答为',output,'答案为',answer)
        total_num += 1
        em = eval_ex_match(output,answer)
        EM += em
        records.append({'index': global_index, 'question': query, 'gold': answer,
                        'pred': output, 'em': bool(em), 'status': 'ok'})
        # LLM_EVAL += LLM_eval(model,query,output,answer)

    if args.output_path:
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump({'model': args.model, 'dataset': args.dataset,
                       'start': args.start, 'end': args.end,
                       'index_file': args.index_file,
                       'answered': total_num, 'total': len(querys),
                       'EM': EM/total_num if total_num else 0.0,
                       'records': records}, f, ensure_ascii=False, indent=2)

    print('answered:', total_num, '/', len(querys))
    print('EM:',EM/total_num if total_num else 0.0)
    # print('LLM EVAL:',LLM_EVAL/total_num)




if __name__ == '__main__':
    main()