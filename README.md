# C code formatting
This repository is a reproduction of the experiment from the paper ["Learning to Format Coq Code Using Language Models"](https://arxiv.org/pdf/2006.16743v1). 

The goal of the project is to train a smaller Transformer-based neural network to predict spacing between tokens in C code.

# Authors
- [Nemanja Vujadinovic](https://github.com/vujadinovicn)

## Modification from the paper
The original paper utilized n-gram models and bi-directional recurrent neural networks to predict spacing and evaluated these models on Coq code. In this project, we made the following modifications:

### Changing the model
Instead of n-gram and RNN models, we opted to use a Transformer-based network. We decided to use the ALBERT transformer. Initially, we considered using the standard BERT model, but due to computational limitations, we opted for a smaller network.

### Changing the dataset
The original study focused on Coq code, whereas this reproduction is trained and evaluated on C code.

## Implementation
Here's how we implemented our C code formatting model:

### Dataset
We use the [GitHub C Code Segmented dataset](https://huggingface.co/datasets/aircrypto/GitHub-C-Code-Segmented), a dataset of segmented C code samples from GitHub repositories. The dataset is loaded and filtered to retain only function-type code samples. Moreover, we reduce the dataset to include only functions with a maximum character length of 100. Then, we further filter these functions to retain only those with 50 or fewer C tokens. In the end, our dataset contained around 40K samples.

### Data pre-processing
To start off, we randomly split our dataset into training, validation and test sets.

Next, we developed a function that uses the Clang library to analyze the provided C code. This function takes each C code sample as a string and breaks it down into tokens. For each token, we capture its name and type—like (such as TokenKind.KEYWORD) and check if there’s a space after it. If there is a space after the token, we append 1 to the space list, else, we append 0. This gives us a more detailed view of how the tokens are structured in the code.

Subsequently, we apply this function to each sample in our dataset. As a result, we created an organized dataset that includes token spellings and types, as an input and spacing information, as our labels. This prepared dataset is now ready for training and evaluation of our model.

### Model
We used AlBERT transformer on top of which we have added a fully connected layer. During the forward pass, we input the information about each token's name/spelling and type into the model. The output from the transformer is then fed into the fully connected layer, where we apply a sigmoid activation function. Output is now transformed into a series of values between 0 and 1, indicating the likelihood of a space following each token. The final output consists of a list of these values, where the number of values corresponds to the number of tokens provided by the tokenizer. 

### Training
Each sample's labels are padded to ensure a consistent length of 50 elements by appending the value 100, which serves as a placeholder. This padding is necessary to standardize the input dimensions across all samples. 

For training, we use binary cross-entropy (BCE) loss. To compute the loss and accuracy accurately, we mask the labels and the model outputs by ignoring the entries where the label is 100. The model is updated thorugh the BCE loss which compares the masked labels, which are either 0s (indicating no space) or 1s (indicating a space), with the corresponding masked outputs from the model, which are also transformed into probabilities between 0 and 1. 

We fully trained our model, meaning that we trained the ALBERT transformer together with the added fully connected layer.

## Results and pretrained model
We have trained our model for 8 epochs. The results can be seen in the table below.
|Validation accuracy | Testing accuracy | Download    |
|:-----------------:|:-------------------:|:-----------:|
|90.10%         | 89.89%           | [link](https://github.com/vujadinovicn/c-formatter-llm/blob/main/checkpoints/space_7.pkl) 


## How to run
The project requires Python 3.8+ and the dependencies listed in requirements.txt. 
Install them using:

```bash
pip install -r requirements.txt
```



### Usage
Dataset files are already prepared and you can find them in `data` directory:
- `reduced_dataset.json`: filtered (by functions and length) dataset
- `train.json`, `val.json`, `test.json`: splitted dataset from `reduced_dataset.json`
- `train_serialized.json`, `val_serialized.json`, `test_serialized.json`: processed and serialized datasets ready for training and evaluation.

However, if you want to generate them yourself, you should do the following:
```
python datasets/reduce_dataset_size.py
python datasets/split_dataset.py
python datasets/reduce_dataset_size.py
```

_NOTE_: If you want to generate dataset on your own, be sure to change the path of `clang` library. You can do this in the 31st line of `datasets/utils.py` script:
``` 
library_path = ".\\{your_venv_name}\\lib\\site-packages\\clang\\native\\libclang.dll" 
```

Next, you can do the training and testing of the model in two ways:
1. Simply run `code_formatting.ipynb` cells. Above every cell, there is the description of its doing. 
2. For training and testing, respectively, you may run:
```
python train.py
```
```
python tester.py
```

_NOTE_: Current configuration sets the `batch_size` to 16. You can search for it (as well as for other hyperparameters) in previously mentioned file and change them.

Enjoy!

# Future improvements
We will add config.yaml file so all the hyperparameters are in one place.
We will add code for inference and formatting of the input.
