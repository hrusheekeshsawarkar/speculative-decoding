"""Tests for the HTTP server's model setup and generation dispatch."""

import asyncio
from types import SimpleNamespace

import pytest
import torch

from speculative_decoding.decoding import DecodingStats
from serving import server


class FakeInputs(dict):
    def to(self, device: torch.device) -> "FakeInputs":
        self.device = device
        return self


class FakeTokenizer:
    bos_token_id = None
    eos_token_id = 1
    pad_token_id = 0
    unk_token_id = 2
    all_special_ids = [0, 1, 2]

    def __init__(self) -> None:
        self.prompt: str | None = None

    def __call__(self, text: str, **kwargs) -> FakeInputs:
        self.prompt = text
        return FakeInputs(input_ids=torch.tensor([[3, 4]]))

    def decode(self, token_ids: list[int], skip_special_tokens: bool) -> str:
        return "decoded"

    def get_vocab(self) -> dict[str, int]:
        return {"<pad>": 0, "</s>": 1, "<unk>": 2, "token": 3}

    def __len__(self) -> int:
        return 4


class FakeModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            is_encoder_decoder=True,
            vocab_size=4,
            decoder_start_token_id=0,
            eos_token_id=1,
        )

    def to(self, device: torch.device) -> "FakeModel":
        return self

    def eval(self) -> "FakeModel":
        return self

    def parameters(self) -> list:
        return []

    def generate(self, **kwargs) -> torch.Tensor:
        return torch.tensor([[0, 3, 1]])


def set_loaded_state(monkeypatch: pytest.MonkeyPatch, tokenizer: FakeTokenizer | None = None) -> FakeTokenizer:
    selected_tokenizer = tokenizer or FakeTokenizer()
    monkeypatch.setattr(
        server,
        "state",
        server.ModelState(
            tokenizer=selected_tokenizer,
            target=FakeModel(),
            draft=FakeModel(),
            device=torch.device("cpu"),
        ),
    )
    return selected_tokenizer


def test_speculative_generate_passes_sampling_options_and_returns_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = set_loaded_state(monkeypatch)
    expected_stats = DecodingStats(iterations=2, drafted=6, accepted=4, rejected=2)
    calls: list[dict] = []

    def fake_decode(**kwargs):
        calls.append(kwargs)
        return [3, 1], expected_stats

    monkeypatch.setattr(server, "speculative_decode", fake_decode)
    response = server.generate(
        server.GenerateRequest(
            text="A text",
            task="summarize",
            temperature=0.7,
            top_k=5,
            top_p=0.9,
            gamma=3,
            seed=42,
        )
    )

    assert tokenizer.prompt == "summarize: A text"
    assert calls[0]["gamma"] == 3
    assert calls[0]["temperature"] == 0.7
    assert calls[0]["top_k"] == 5
    assert calls[0]["top_p"] == 0.9
    assert response.token_ids == [3, 1]
    assert response.num_new_tokens == 2
    assert response.accepted_tokens == 4
    assert response.drafted_tokens == 6
    assert response.iterations == 2
    assert response.acceptance_rate == pytest.approx(2 / 3)


def test_autoregressive_mode_keeps_hugging_face_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    set_loaded_state(monkeypatch)
    response = server.generate(
        server.GenerateRequest(text="input", mode="autoregressive", temperature=0.8, top_k=7, top_p=0.95)
    )

    assert response.mode == "autoregressive"
    assert response.token_ids == [3, 1]
    assert response.iterations is None
    assert response.drafted_tokens is None


def test_model_compatibility_rejects_different_token_ids() -> None:
    tokenizer = FakeTokenizer()
    other_tokenizer = FakeTokenizer()
    other_tokenizer.get_vocab = lambda: {"<pad>": 0, "</s>": 1, "<unk>": 2, "other": 3}

    with pytest.raises(ValueError, match="identical token IDs"):
        server.validate_compatible_models(FakeModel(), FakeModel(), tokenizer, other_tokenizer)


def test_lifespan_loads_and_clears_both_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server.torch.cuda, "is_available", lambda: False)
    tokenizer_loads: list[str] = []
    model_loads: list[str] = []

    def load_tokenizer(cls, model_name: str) -> FakeTokenizer:
        tokenizer_loads.append(model_name)
        return FakeTokenizer()

    def load_model(cls, model_name: str, dtype: torch.dtype) -> FakeModel:
        model_loads.append(model_name)
        return FakeModel()

    monkeypatch.setattr(server.AutoTokenizer, "from_pretrained", classmethod(load_tokenizer))
    monkeypatch.setattr(server.T5ForConditionalGeneration, "from_pretrained", classmethod(load_model))

    async def run_lifespan() -> None:
        async with server.lifespan(server.app):
            assert server.state is not None
            assert isinstance(server.state.target, FakeModel)
            assert isinstance(server.state.draft, FakeModel)

    asyncio.run(run_lifespan())

    assert tokenizer_loads == [server.MODEL_NAME, server.DRAFT_MODEL_NAME]
    assert model_loads == [server.MODEL_NAME, server.DRAFT_MODEL_NAME]
    assert server.state is None
