import torch
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingManager:
    def __init__(self, model_name: str ="Qwen/Qwen3-Embedding-0.6B"):
        print(f"[Embedding] Loading model : {model_name}")
        self.model = SentenceTransformer(
            model_name,
            model_kwargs={
                "device_map": "auto",
                "torch_dtype": "auto",
            },
            tokenizer_kwargs={"padding_side": "left"},
        )
        

    def embed_text(self,text: str, command: str = "document") -> np.ndarray:

        if not text.strip():
            print(f"There is no text")
            return np.zeros(1024, dtype=np.float32)
        
        emb = self.model.encode(
            [text],
            prompt_name = command, 
            normalize_embeddings=True,
        )[0]
        return emb.astype(np.float32)