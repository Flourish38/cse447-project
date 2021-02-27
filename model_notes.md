# Models

Name | Description
--- | ---
`test1` | Neural LSTM encoder with all embedding size 100, trained over toy data
`model-2` | Neural LSTM encoder, hidden size 100, trained over downsample data
`model-3` | Same as `model-2` , but hidden size 200
`model-3-c` | Continued training of `model-3`, many more epochs, over randomly downsampled data in groups of 3 epochs
`model-4` | Same as `model-3`, but num_layers 2
`model-4-c` | Continued training of `model-4`, like `model-3-c`, but downsampled data takes 2 epochs
