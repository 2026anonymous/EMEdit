import torch


class FrozenQwenTextEncoder:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-4B",
        device: str | None = None,
    ):
        from sentence_transformers import SentenceTransformer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()

    @torch.no_grad()
    def encode(self, texts, batch_size: int = 32) -> torch.Tensor:
        features = self.model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return features.detach().float()
