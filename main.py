from rag import *
from llm import *
from utils import *
from attack import *
from pipeline import *


def main():
    
    
    config = get_config()
    config.attacker = get_attacker(config=config)
    config.rag = get_rag(config=config)
    config.llm = get_llm(config=config)
    
    # TODO: Load workflow from outside
    workflow_4_poison_rag = [
        # 'select',
        # 'poison', 
        'prerag', 
        'evaluate', 
    ]
    workflow_4_normal_rag = ['preprocess', 'prerag']
    if config.normal_eval:
        workflow_4_normal_rag = ['evaluate']
    
    workflow = workflow_4_normal_rag if config.normal or config.normal_eval else workflow_4_poison_rag
    
    pipeline = PipeLine(workflow=workflow, config=config)
    pipeline.run_workflow()
    
    


if __name__ == "__main__":
    main()