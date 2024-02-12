import enum

from langchain.llms import HuggingFacePipeline
from ctransformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline



CACHE_DIR = './cache'
CHROMA_DIR = './chroma'


class Models:
    FLAN_LARGE = "google/flan-t5-large" # 800M
    MISTRAL = "TheBloke/Mistral-7B-OpenOrca-GGUF" # 7B
    @staticmethod
    def get_llm(model: str):
        if model == Models.FLAN_LARGE:
            model = HuggingFacePipeline.from_model_id(
                model_id=model,
                task="text2text-generation",
                model_kwargs={
                    "temperature": 0,
                    "max_length": 128,
                    "cache_dir": CACHE_DIR,
                },
            )
            return model
        elif model == Models.MISTRAL:
            # for more inspiration look at:
            # https://huggingface.co/TheBloke/MPT-7B-Instruct-GGML/discussions/2
            model_id = model
            llm = AutoModelForCausalLM.from_pretrained(model_id)
            return llm
            # llm = AutoModelForCausalLM.from_pretrained(model_id, hf=True)
            # tokenizer = AutoTokenizer.from_pretrained(llm)
            # pipe = pipeline("text2text-generation", model=llm, tokenizer=tokenizer)
            # return pipe
            # let's just use the regular embeddings for now
            # embeddings = LlamaCppEmbeddings(model_path="models/llama-7b.ggmlv3.q4_0.bin")
        else:
            raise Exception(f'model {model} not supported')
