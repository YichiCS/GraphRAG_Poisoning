import os
import re
import json
import random
import traceback
from prompt import prompt

from tqdm import tqdm
from pathlib import Path
from utils import save_json, load_json

# Using HuggingFace
from datasets import load_dataset

__all__ = ['PreProcess', 'Select', 'Poison', 'PreRAG', 'Evaluate']
    
class WorkFlow():
    def __init__(self, config, logger) -> None:
        self.config = config
        self.logger = logger
        
    def run(self, **kwargs):
        # TODO: return code
        raise NotImplementedError
    
    def resume(self, path):
        if os.path.exists(path):
            result = load_json(path)
            id = [item['id'] for item in result]
            info = f'[{self.__class__.__name__}]: Load Existing Data from {path}'
        else:
            result = []
            id = []
            info = f'[{self.__class__.__name__}]: No Existing Data in {path}'
        self.logger.info(info)
        return result, id
        
        
    
class PreProcess(WorkFlow):
    dataset_dict = {
        'squad': 'rajpurkar/squad',
        'squad_v2': 'rajpurkar/squad_v2', 
        # 'nq': 'google-research-datasets/natural_questions',
        # 'ms_marco': 'microsoft/ms_marco',
    }
    def __init__(self, config, logger) -> None:
        super().__init__(config, logger)
        
    def run(self, **kwargs):
        try:
            config = self.config
            dataset_name = PreProcess.dataset_dict[config.dataset]
            if config.dataset in ['squad', 'squad_v2']:
                if os.path.exists(config.clean_path):
                    return self.logger.info(f'[{self.__class__.__name__}]: Clean Dataset has been stored in {config.clean_path}')
                dataset = load_dataset(dataset_name, split=config.split)
                
                titles, document = [], []
                for item in tqdm(dataset, desc=f'Loading {dataset_name}', ncols=config.ncol):
                    if item['answers']['text'] == []:
                        continue    # TODO: [] in Squad V2
                    if item['title'] not in titles:
                        document.append({
                            'id': len(document), 
                            'title': item['title'],
                            'corpus': [item['context']], 
                            'queries': [{
                                'id': 0,
                                'question': item['question'],
                                'answer':   item['answers']['text'][0], # load the firstr answer
                            }],
                        })
                        titles.append(item['title'])
                    else:
                        idx = titles.index(item['title'])
                        document[idx]['queries'].append({
                                'id': len(document[idx]['queries']),
                                'question': item['question'],
                                'answer':   item['answers']['text'][0],
                            })
                        if item['context'] not in document[idx]['corpus']:
                            document[idx]['corpus'].append(item['context'])
                        
                for i in range(len(document)):
                    document[i]['corpus'] = '\n\n'.join(document[i]['corpus'])
                if config.mini > 0:
                    document = document[:config.mini]    
                
                save_json(document, config.clean_path)
            else:
                raise ValueError('Unexpected Dataset {dataset}'.format(config.dataset))
            
            return self.logger.info(f'[{self.__class__.__name__}]: Clean Dataset has been stored in {config.clean_path}')
        
        except:
            tb_info = traceback.format_exc()
            return self.logger.info(f'[{self.__class__.__name__}]: {tb_info}')


class Select(WorkFlow):
    def __init__(self, config, logger) -> None:
        super().__init__(config, logger)
        
    def run(self, **kwargs):
        try:
            config = self.config
            rag = config.rag
            
            clean_data = load_json(config.clean_path)
            selected_result, selected_id = self.resume(config.select_path)
            random.seed(config.seed)
            
            for document in clean_data:
                d_id = document['id']
                queries = document['queries']
                random.shuffle(queries)
                sub_rag_dir = os.path.join(config.clean_rag_dir, f'document_{d_id}')
                
                if d_id in selected_id or not os.path.exists(sub_rag_dir):
                    continue
                
                sucess_query = []
                success_cnt = 0
                tbar = tqdm(total=config.query_max, ncols=config.ncol, desc=f'Document {d_id} success_query 0/0')
                for query in queries:
                    answer = query['answer']
                    question = query['question']
                    response = rag.query(sub_rag_dir, question)
                    query['response'] = response
                    if answer.lower() in response.lower():
                        query['result'] = True
                        success_cnt = success_cnt + 1 
                        tbar.update(1)
                    else:
                        query['result'] = False
                    sucess_query.append(query)
                    tbar.set_description(f'Document {d_id} success_query {success_cnt}/{len(sucess_query)}')
                    if success_cnt >= config.query_max:
                        break
                selected_result.append({
                    "id": d_id, 
                    "acc": success_cnt/len(sucess_query), 
                    "queries": sucess_query
                })
                save_json(selected_result, config.select_path)
            return self.logger.info(f'[{self.__class__.__name__}]: Selected Dataset has been stored in {config.select_path}')
        except:
            tb_info = traceback.format_exc()
            return self.logger.info(f'[{self.__class__.__name__}]: {tb_info}')

        
class Poison(WorkFlow):
    def __init__(self, config, logger) -> None:
        super().__init__(config, logger)
        
    def run(self, **kwargs):
        try:
            config = self.config
            
            selected_data = load_json(config.select_path)
            poisoned_result, poisoned_id = self.resume(config.poisoned_path)
            
            for document in tqdm(selected_data, desc=f'Poisoning with {config.attack}', ncols=config.ncol):
                d_id = document['id']
                if d_id in poisoned_id:
                    continue
                queries = [q for q in document['queries'] if q['result']]
                random.shuffle(queries)
                selected_queries = random.sample(queries, config.query_num)
                selected_answers = [{
                        'id': q['id'], 
                        'answer': q['answer'], 
                    } for q in selected_queries]
                replaced_answers = self.replace_answer(selected_answers, config)
                
                if replaced_answers is []:
                    self.logger.warning(f"[{self.__class__.__name__}]: Fail to replace answer for Document {d_id}")
                    continue
                
                replaced_queries = [{
                    'id': q['id'], 
                    'question': q['question'],
                    'answer': a['replaced_answer'], 
                } for q, a in zip(selected_queries, replaced_answers)]
                
                for idx, query in enumerate(replaced_queries):
                    q_id = query['id']
                    corpus = self.generated_corpus(query, config)
                    if corpus == []:
                        self.logger.warning(f"Fail to corpus for Query {q_id}")
                    selected_queries[idx]['replaced_answer'] = replaced_answers[idx]['replaced_answer']
                    selected_queries[idx]['corpus'] = corpus
                    del selected_queries[idx]['result'] 
                    del selected_queries[idx]['response'] 
                    
                poisoned_result.append({
                    'id':d_id, 
                    'queries':selected_queries
                })
                save_json(poisoned_result, config.poisoned_path)
            return self.logger.info(f'[{self.__class__.__name__}]: Poisoned Dataset has been stored in {config.select_path}')
        except:
            tb_info = traceback.format_exc()
            return self.logger.info(f'[{self.__class__.__name__}]: {tb_info}')

    def replace_answer(self, answers, config):
        attempt_count = 0
        while attempt_count <= config.max_attempts:
            attempt_count += 1
            try:
                response = config.llm.generate(
                    model=config.llm_attack_random,
                    prompt=prompt['answer_replace']['user'].format(k=len(answers), answers=answers),
                    system=prompt['answer_replace']['system'],
                )
                replaced_answers = json.loads(response['response'])['result']
                if len(replaced_answers) == len(answers):
                    return replaced_answers
            except Exception as error:
                pass
        return []

    def generated_corpus(self, query, config):
        attempt_count = 0
        corpus = []
        while attempt_count < config.max_attempts:
            try:
                response = config.llm.generate(
                    model=config.llm_attack_random,
                    prompt=prompt['poison_generation']['user'].format(pairs=query,  num=config.corpus_len),
                    system=prompt['poison_generation']['system'],
                )
                _corpus = json.loads(response['response'])['corpus']
                if self.answer_check(_corpus, query['answer']):
                    corpus.append(_corpus)
                if len(corpus) == config.corpus_num:
                    return corpus
                attempt_count = attempt_count + 1
            except Exception as error:
                pass
        return []

    def answer_check(self, corpus, answer):
        # TODO
        target = answer.split()
        for t in target:
            if t in corpus:
                return True
        return False
    
class PreRAG(WorkFlow):
    def __init__(self, config, logger) -> None:
        super().__init__(config, logger)
    # TODO: Check RAG
    def skip_check(self, root, config):
        if config.overwrite:
            return False, False
        # TODO input file check
        if os.path.exists(os.path.join(root, 'input', 'clean.txt')): 
            path = os.path.join(root, 'output')
            if not os.path.exists(path):
                return True, False
            path = sorted(Path(path).iterdir(), key=os.path.getmtime, reverse=True)[0]
            file_num = len(os.listdir(os.path.join(path, 'artifacts')))
            if file_num == 23: # TODO ourput file check
                return True, True
            else:
                return True, False
        else:
            return False, False
        
    def run(self, **kwargs):
        try:
            config = self.config
            rag = config.rag
            
            clean_dataset = load_json(config.clean_path)
            # TODO： when attack is 'None'
            poisoned_dataset = clean_dataset if config.normal else load_json(config.poisoned_path)
            rag_result, rag_id = self.resume(config.rag_path)
            
            for document in poisoned_dataset:
                d_id = document['id']
                sub_rag_dir = os.path.join(config.rag_dir, f"document_{d_id}")
                # TODO Skip
                skip_0, skip_1 = self.skip_check(sub_rag_dir, config)
                if not skip_0:
                    os.makedirs(os.path.join(sub_rag_dir, 'input'), exist_ok=True)
            
                    with open(os.path.join(sub_rag_dir, 'input', 'clean.txt'), 'w') as file:
                        file.write(clean_dataset[d_id]["corpus"])
                        
                    if config.attack == 'poisoned':
                        for query in document['queries']:
                            q_id = query['id']
                            question = query['question']
                            
                            # TODO: Data Type Checking
                            corpus = query['corpus'][:min(config.rag_corpus_num, len(query['corpus']))]
                            
                            p_corpus = '\n\n'.join([question + ' ' + _c for _c in corpus if isinstance(_c, str)])
                            
                            with open(os.path.join(sub_rag_dir, 'input', f'poisoned_{q_id}.txt'), 'w') as file:
                                file.write(p_corpus)
                    elif config.attack == 'none':
                        pass
                
                if not skip_1:
                    rag.init(root=sub_rag_dir, config=config.setting)
                    rag_result.append({
                        'id': d_id, 
                        'title': clean_dataset[d_id]['title'], 
                        'rag': True,
                    })
                    
                self.logger.info(f'[{self.__class__.__name__}]: Success GraphRAG.INDEX Document {d_id}')
                save_json(rag_result, config.rag_path)
                
            return self.logger.info(f'[{self.__class__.__name__}]: Environment has been created in {config.select_path}')
        except:
            tb_info = traceback.format_exc()
            return self.logger.info(f'[{self.__class__.__name__}]: {tb_info}')
        
    def resume(self, path):
        if os.path.exists(path):
            result = load_json(path)
            id = [item['id'] for item in result if item['rag']]
            info = f'[{self.__class__.__name__}]: Load Existing Data from {path}'
        else:
            result = []
            id = []
            info = f'[{self.__class__.__name__}]: No Existing Data in {path}'
        self.logger.info(info)
        return result, id
    
class Evaluate(WorkFlow):
    def __init__(self, config, logger) -> None:
        super().__init__(config, logger)
        
    def run(self, **kwargs):
        try:
            config = self.config
            rag = config.rag
            llm = config.llm
    
            rag_data = load_json(config.rag_path)
            poisoned_dataset = load_json(config.poisoned_path)
            eval_result, eval_id, = self.resume(config.eval_path)
            # TODO: Skip
            for document in poisoned_dataset:
                d_id = document['id']
                if not rag_data[d_id]['rag'] or d_id in eval_id:
                    continue
                sub_rag_dir = os.path.join(config.rag_dir, f"document_{d_id}")
                for query in tqdm(document['queries'], ncols=80, desc=f'Document {d_id}'):
                    question = query['question']
                    response = rag.query(sub_rag_dir, question, config.setting_eval)
                    
                    query['document_id'] = d_id
                    query['response'] = response
                    del query['corpus']
                            
                    eval_result.append(query)
                save_json(eval_result, config.eval_path)
                
            for idx, query in enumerate(tqdm(eval_result, ncols=80, desc=f'Checking Answers')):
                if 'attack_result' in query.keys() and 'normal_result' in query.keys():
                    continue
                # TODO: PostgAnswerCheck
                question = query['question']
                attack_result = self.answer_check(question, query['response'], query['replaced_answer'], llm, config)
                normal_result = self.answer_check(question,query['response'], query['answer'], llm, config)
                query['attack_result'] = attack_result
                query['normal_result'] = normal_result
                
                if idx % 10 == 0:
                    save_json(eval_result, config.eval_path)
            save_json(eval_result, config.eval_path)
            
            # TODO Metric
            asr_cnt, csr_cnt, bsr_cnt = 0, 0, 0
            for query in eval_result:
                ar = query['attack_result']['overall']
                cr = query['normal_result']['overall']
                
                if ar and (cr in [None, False]): asr_cnt += 1
                elif cr and (ar in [None, False]): csr_cnt += 1
                else: bsr_cnt += 1 
            self.logger.info(f'Attack: {asr_cnt}/{len(eval_result)} = {100*asr_cnt/len(eval_result):.2f}% \nClean: {csr_cnt}/{len(eval_result)} = {100*csr_cnt/len(eval_result):.2f}% \nBalance: {bsr_cnt}/{len(eval_result)} = {100*bsr_cnt/len(eval_result):.2f}%')
                        
            return self.logger.info(f'[{self.__class__.__name__}]: Evaluation Result has been stored in {config.rag_path}')
        except:
            tb_info = traceback.format_exc()
            return self.logger.info(f'[{self.__class__.__name__}]: {tb_info}')
    
    def resume(self, path):
        if os.path.exists(path):
            result = load_json(path)
            id = [item['document_id'] for item in result]
            info = f'[{self.__class__.__name__}]: Load Existing Data from {path}'
        else:
            result = []
            id = []
            info = f'[{self.__class__.__name__}]: No Existing Data in {path}'
        self.logger.info(info)
        return result, id
            
    def answer_check(self, question, response, answer, llm, config):
        split = False
        _response = response.lower().split()
        for w in _response:
            if w in answer.lower().split():
                split = True
                break
        if not split:
            return {
                'overall': False, 
                'split_match': split,
                'exact_match': None, 
                'llm_match': None,
            }
        
        # TODO: Exact Match
        pattern = r'\[[^\]]+\]'
        exact = True if re.sub(pattern, '', response.lower()).strip() == answer.lower() else False
        if exact:
            return {
                'overall': True, 
                'split_match': split,
                'exact_match': exact, 
                'llm_match': None,
            }
        
        query = {
            "corpus": response.replace("\"", ""), 
            "question": question, 
            "answer": answer,
        }
        try:
            response = llm.generate(
                model=config.llm_user,
                prompt=prompt['answer_check']['user'].format(query=query),
                system=prompt['answer_check']['system'],
            )
            llm_ = json.loads(response['response'])['result']
            assert llm_ in [True, False]

            return {
                'overall': llm_, 
                'split_match': split,
                'exact_match': exact, 
                'llm_match': llm_,
            }
        
        except Exception as error:
            return {
                'overall': None, 
                'split_match': split,
                'exact_match': exact, 
                'llm_match': None,
            }
        except KeyboardInterrupt:
            exit()