

__all__ = ['get_llm']

def get_llm(config):
    llm_dict = {
        'ollama': OllamaLLM, 
    }
    return llm_dict[config.llm_type](config)


class LLM():
    def __init__(self, config) -> None:
        self.config = config
        
    def generate(self, **kwargs):
        # TODO: return code
        raise NotImplementedError

class OllamaLLM(LLM):
    def __init__(self, config) -> None:
        super().__init__(config)
        
    def generate(self, model, prompt, system='', format='json',  **kwargs):
        import ollama
        response = ollama.generate(
            model=model, 
            format=format,
            system=system, 
            prompt=prompt
        )
        return response
    