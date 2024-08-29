import os
import shutil
import subprocess

__all__ = ['get_rag']

def get_rag(config):
    rag_dict = {
        'graphrag': GraphRAG, 
    }
    return rag_dict[config.rag](config)


class RAG():
    def __init__(self, config) -> None:
        self.config = config
        
    def init(self, **kwargs):
        # TODO: return code
        raise NotImplementedError
        
    def query(self, **kwargs):
        # TODO: return code
        raise NotImplementedError
        
    def eval(self, **kwargs):
        # TODO: return code
        raise NotImplementedError
    
class GraphRAG(RAG):
    # TODO: Action Dict
    init_cmd  = 'python -m graphrag.index --root {root} --reporter print --init'
    index_cmd = 'python -m graphrag.index --root {root} --reporter print'
    query_cmd = 'python -m graphrag.query --root {root} --method local "{question}" --response_type "A word or phrase that contains the necessary answer, as simple as possible"'
    eval_cmd = 'python -m graphrag.query --root {root} --method local "{question}" --config {config} --response_type "A word or phrase that contains the necessary answer, as simple as possible"'
    
    def __init__(self, config) -> None:
        super().__init__(config)
        
    def init(self, root, config, **kwargs):
        # TODO: Check File
        
        init_result = subprocess.run([
            GraphRAG.init_cmd.format(root=root),
            ], shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        shutil.copy(config, os.path.join(root, 'settings.yaml'))
        
        index_result = subprocess.run([
            GraphRAG.index_cmd.format(root=root),
            ], shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        
        return init_result, index_result
        
    def query(self, root, question, config=None, **kwargs):
        cmd = GraphRAG.eval_cmd.format(root=root, question=question, config=config) if config else GraphRAG.query_cmd.format(root=root, question=question)
        
        # TODO: response check
        response = subprocess.run([cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        response = response.stdout.decode("utf-8")
        response = response[response.find("SUCCESS: Local Search Response:") + len("SUCCESS: Local Search Response:"):].strip()
        return response