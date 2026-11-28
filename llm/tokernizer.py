#!/usr/bin/python3
"""
This script is meant to understand how tokens are created and tokenizers assign
numerical values to the token.
References:
1. https://huggingface.co/ibm-granite/granite-4.2-3b/raw/main/tokenizer.json
2. https://huggingface.co/ibm-granite/granite-4.2-3b/blob/main/tokenizer_config.json
3. https://huggingface.co/ibm-granite/granite-4.2-3b/tree/main

Usage:
    uv run tokernizer.py --tokenize
    uv run tokernizer.py --compare
    uv run tokernizer.py --architecture
    uv run tokernizer.py --embeddings
"""

from optparse import OptionParser
from transformers import AutoTokenizer, AutoConfig

model = "ibm-granite/granite-4.2-3b"

tokenizer = AutoTokenizer.from_pretrained(model)


def demo_tokenize_and_decode():
    """Demonstrate the full tokenization roundtrip for a single sentence.

    Takes a sentence and walks through each step:
      text → sub-word tokens → token IDs → decoded text
    Also prints the tokenizer's vocabulary size.
    """
    text = "What is the status of my Kubernetes?"

    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text)

    print("Tokens:", tokens)
    print("Token IDs:", token_ids)

    print("Decoded:", tokenizer.decode(token_ids))

    print(tokenizer.vocab_size)
    # print(tokenizer.special_tokens_map)


def compare_tokenization():
    """Compare how different texts are split into tokens.

    Shows how casing, repetition, and multi-word inputs affect
    tokenization — e.g. lowercase "kubernetes" vs titlecase "Kubernetes"
    may produce different sub-word splits and token counts.
    """
    texts = [
        "kubernetes",
        "Kubernetes Kubernetes",
        "Kubernetes Cluster status"
    ]
    for text in texts:
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text)
        print(text, "-->", tokens, '-->', len(tokens), '-->', token_ids)


def print_model_architecture():
    """Print the model's architecture details from its HuggingFace config.

    Loads the config for granite-4.2-3b and prints vocabulary size,
    hidden size (embedding dimension), number of transformer layers,
    and number of attention heads.
    """
    config = AutoConfig.from_pretrained(model)
    print("Vocabulary Size: ", config.vocab_size)
    print("Hidden Size: ", config.hidden_size)
    print("Layers: ", config.num_hidden_layers)
    print("Attention Heads: ", config.num_attention_heads)


def extract_token_embeddings():
    """Extract and display the embedding vector for each token in a word.

    Uses llama_cpp to load a local GGUF model file, tokenizes the input,
    then retrieves the actual numerical embedding vectors the model uses
    internally. Prints each token's ID, string representation, vector
    length, and the first 5 dimensions of its embedding.
    """
    from llama_cpp import Llama

    # Point to your local Q4_K_M .gguf file
    llm = Llama(
        model_path="/Users/X/.ollama/models/blobs/sha256-20e436143017578687f7f848225cc6c6038126c84149192229c7dff6e4e0f427",
        embedding=True,   # Enable embedding extraction
        verbose=False
    )

    text = "Kubernetes"

    # 1. See tokens & IDs
    token_ids = llm.tokenize(text.encode("utf-8"))
    print("Token IDs:", token_ids)

    # 2. Get embeddings (returns a vector per token when pooling_type is None/0)
    embeddings = llm.create_embedding(text)
    token_vectors = embeddings["data"][0]["embedding"]

    print("Number of token vectors:", len(token_vectors))         # 2
    print("Embedding dimension per token:", len(token_vectors[0])) # e.g., 2048 or 2560

    for tid, vec in zip(token_ids, token_vectors):
        tok_str = llm.detokenize([tid]).decode("utf-8", errors="replace")
        print(f"Token '{tok_str}' (ID {tid}) -> length {len(vec)}, first 5: {vec[:5]}")


if __name__ == "__main__":
    parser = OptionParser(usage="usage: %prog [options]")
    parser.add_option(
        "-t", "--tokenize",
        action="store_true", dest="tokenize", default=False,
        help="Run the tokenization roundtrip demo (text → tokens → IDs → decoded text)"
    )
    parser.add_option(
        "-c", "--compare",
        action="store_true", dest="compare", default=False,
        help="Compare tokenization across different texts (casing, repetition)"
    )
    parser.add_option(
        "-a", "--architecture",
        action="store_true", dest="architecture", default=False,
        help="Print model architecture details (vocab size, hidden size, layers, heads)"
    )
    parser.add_option(
        "-e", "--embeddings",
        action="store_true", dest="embeddings", default=False,
        help="Extract and display token embedding vectors from a local GGUF model"
    )

    (options, args) = parser.parse_args()

    # Track whether any option was selected
    ran = False

    if options.tokenize:
        demo_tokenize_and_decode()
        ran = True
    if options.compare:
        compare_tokenization()
        ran = True
    if options.architecture:
        print_model_architecture()
        ran = True
    if options.embeddings:
        extract_token_embeddings()
        ran = True

    if not ran:
        parser.print_help()
