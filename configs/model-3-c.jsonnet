{
    "distributed": {
        "cuda_devices": [0, 1]
    },
    "vocabulary": {
        "non_padded_namespaces": [],
    },
    "datasets_for_vocab_creation": ["train"],
    "dataset_reader": {
        "type": "my_data",
        "max_tokens": 600,
        "data_root": "/local1/d0/447-data"
    },
    "train_data_path": "train_new",
    "model": {
        "type": "custom_lm",
        "embedder": {
            "token_embedders": {
                "tokens": {
                    "embedding_dim": 200
                }
            }
        },
        "encoder": {
            "type": "lstm",
            "input_size": 200,
            "hidden_size": 200
        }
    },
    "data_loader": {
        "batch_size": 32,
        "shuffle": true
    },
    "trainer": {
        "optimizer": "adam",
        "num_epochs": 6,
        "validation_metric": "-perplexity"
    },
    "validation_data_path": "dev_down",
    "validation_data_loader": {
        "batch_size": 8,
        "shuffle": false
    }
}