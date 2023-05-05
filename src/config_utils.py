# Hugging Face Style Model Configuration

from transformers.configuration_utils import PretrainedConfig


class PerceiverARConfig(PretrainedConfig):
    """
    """

    model_type = "perceiverAR"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        vocab_size=1517,
        n_positions=1024,  # seq_len
        n_embd=768,
        n_layer=8,
        n_head=12,
        bos_token_id=1,
        eos_token_id=1,
        initializer_range=.02,
        pdrop=.1,
        # resid_pdrop=.1,
        # embd_pdrop=.1,
        # attn_pdrop=.1,
        residual=True,
        pe_type='rotary',
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.residual = residual
        self.initializer_range = initializer_range
        self.pdrop = pdrop
        # self.resid_pdrop = resid_pdrop
        # self.embd_pdrop = embd_pdrop
        # self.attn_pdrop = attn_pdrop

        self.pe_type = pe_type

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
