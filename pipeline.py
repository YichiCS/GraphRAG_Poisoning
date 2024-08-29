import os
import logging
import traceback
from datetime import datetime

from workflow import *

__all__ = ['PipeLine']


class PipeLine():
    log = '[Workflow]: {name}\t[Return Code]:{return_code} \n[Info]: {info}'
    workflow_dict = {
        'preprocess': PreProcess,
        'select': Select, 
        'poison': Poison, 
        'prerag': PreRAG, 
        'evaluate': Evaluate, 
    }
    
    def __init__(self, workflow, config) -> None:
        self.config = config
        self.logger = self.get_logger(config)
        self.workflow = self.get_workflow(workflow)
        
        self.logger.info(
            f'Running Workflows {workflow}'
        )
        
    def run_workflow(self, **kwargs):
        try:
            for workflow in self.workflow:
                workflow.run()
        except:
            tb_info = traceback.format_exc()
            return self.logger.info(f'[{self.__class__.__name__}]: {tb_info}')
            
    def get_workflow(self, workflow):
        return [PipeLine.workflow_dict[w](self.config, self.logger) for w in workflow]
    
    def get_logger(self, config):
        log_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_name = f'{log_time}.log'
        log_file_path = os.path.join(config.log_dir, log_name)
        
        logger = logging.getLogger('logger')
        logger.setLevel(logging.DEBUG)  
        
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter('[Logger]: %(asctime)s - %(name)s - %(levelname)s \n%(message)s \n', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger