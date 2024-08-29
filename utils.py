import os
import json
import random
import argparse

__all__ = [
    'get_config', 
]

def seed_everything(seed):
    random.seed(seed)
    # TODO: Not Used
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

def get_config():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", default='squad', type=str, choices=['squad', 'squad_v2'])
    parser.add_argument("--split", default='validation', type=str, choices=['validation'])
    parser.add_argument("--attack", default='poisoned', type=str, choices=['poisoned', 'none'])
    
    parser.add_argument("--data_dir", default='./.data', type=str)
    parser.add_argument("--result_dir", default='./.result', type=str)
    parser.add_argument("--setting_dir", default='./.setting', type=str)
    parser.add_argument("--setting", default='./.setting/settings_index.yaml', type=str)
    parser.add_argument("--setting_eval", default='./.setting/settings_eval.yaml', type=str)
    parser.add_argument("--experiment_dir", default='./.experiment', type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--ncol", default=80, type=int)
    
    # RAG
    parser.add_argument("--rag", default='graphrag', type=str, choices=['graphrag'])
    # TODO: Attacker
    # TODO: Small ctx used in Index, Long in Query
    # LLM
    parser.add_argument("--llm_type", default='ollama', type=str, choices=['ollama'])
    parser.add_argument("--llm_attack", default='llama3-16k-42', type=str)
    parser.add_argument("--llm_user", default='mistral-16k-42', type=str)
    parser.add_argument("--llm_attack_random", default='llama3', type=str)
    
    # select
    parser.add_argument("--query_max", default=10, type=int)
    # poisoned
    parser.add_argument("--query_num", default=10, type=int)
    parser.add_argument("--max_attempts", default=20, type=int)
    parser.add_argument("--corpus_num", default=5, type=int)
    parser.add_argument("--corpus_len", default=30, type=int)
    # prerag
    parser.add_argument("--overwrite", default=False, type=bool)
    parser.add_argument("--rag_corpus_num", default=5, type=int)
    
    # Debug Option YICHI
    parser.add_argument("--normal", action='store_true')
    parser.add_argument("--mini", default=-1, type=int)
    
    args = parser.parse_args()
    if args.normal:
        args.attack = 'none'
    if args.mini > 0:
        args.data_split = f"{args.dataset}_{args.split}_mini_{args.mini}"
    else:
        args.data_split = f"{args.dataset}_{args.split}"
    args.exp_name = f"{args.data_split}_{args.attack}"
    
    args.log_dir = os.path.join(args.result_dir, args.exp_name, 'log')
    os.makedirs(args.log_dir, exist_ok=True)
    
    args.rag_dir = os.path.join(args.experiment_dir, args.exp_name)
    os.makedirs(args.rag_dir, exist_ok=True)
    args.clean_rag_dir = os.path.join(args.experiment_dir, args.data_split+'_none')
    
    args.clean_path = os.path.join(args.data_dir, f'{args.data_split}.json')
    args.select_path = os.path.join(args.result_dir, args.exp_name, 'select_result.json')
    args.poisoned_path = os.path.join(args.result_dir, args.exp_name, 'poisoned_result.json')
    args.rag_path = os.path.join(args.result_dir, args.exp_name, 'rag_result.json')
    args.eval_path = os.path.join(args.result_dir, args.exp_name, 'eval_result.json')
    # TODO: unused
    # args.device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(args.seed)
        
    return args
    

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    # TODO
    return True

def load_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data