from rag import *
from llm import *
from utils import *
from pipeline import *


def main():
    
    # TODO: Load workflow from outside
    workflow_4_poison_rag = ['select','poison', 'prerag', 'evaluate']
    workflow_4_normal_rag = ['preprocess', 'prerag']
    
    config = get_config()
    config.rag = get_rag(config=config)
    config.llm = get_llm(config=config)
    
    workflow = workflow_4_normal_rag if config.normal else workflow_4_poison_rag
    
    pipeline = PipeLine(workflow=workflow, config=config)
    pipeline.run_workflow()
    
    


if __name__ == "__main__":
    main()