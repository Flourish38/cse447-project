{
    "distributed": {
        "cuda_devices": [0, 1]
    },
    "datasets_for_vocab_creation": ["train"],
    "dataset_reader": {
        "type": "my_data",
        "max_tokens": 600,
        "data_root": "/local1/d0/447-data"
    },
    "train_data_path": "train",
    "model": {
        "type": "custom_lm",
        "embedder": {
            "token_embedders": {
                "tokens": {
                    "embedding_dim": 100
                }
            }
        },
        "encoder": {
            "type": "lstm",
            "input_size": 100,
            "hidden_size": 100
        }
    },
    "data_loader": {
        "batch_size": 16,
        "shuffle": true
    },
    "trainer": {
        "optimizer": "adam",
        "num_epochs": 10,
        "validation_metric": "-perplexity"
    },
    "validation_data_path": "dev",
    "validation_data_loader": {
        "batch_size": 8,
        "shuffle": false
    }
}