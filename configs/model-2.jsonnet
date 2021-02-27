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
        "data_root": "/local1/d0/447-data",
        "black_list": ["all_talks_train.txt"]
    },
    "train_data_path": "train_down",
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
        "num_epochs": 3,
        "validation_metric": "-perplexity"
    },
    "validation_data_path": "dev_down",
    "validation_data_loader": {
        "batch_size": 8,
        "shuffle": false
    }
}