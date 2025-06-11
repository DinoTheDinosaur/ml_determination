# ml_determination
A package for determining the matrix language in bilingual sentences. This is the implementation of the algorithms presented in the paper [Methods for Automatic Matrix Language Determination of Code-Switched Speech](https://aclanthology.org/2024.emnlp-main.330/). Currently supports English/Mandarin code-switching, create a [feature request](https://github.com/DinoTheDinosaur/ml_determination/issues/new/choose) if you want the system to be extended to other languages.


## Installation

The main functionality can be easily installed into your Python environment using pip:

```shell
pip install ml-determination
```


## Usage

To predict the matrix language using the package import the library and the matrix language determination classes for text:

```python
>>> from ml_determination.predict_matrix_language import MatrixLanguageDeterminerWordMajority
>>> ml = MatrixLanguageDeterminerWordMajority(L1='ZH', L2='EN')
>>> ml.determine_ML('然后 那些 air supply 的 然后 michael learns to rock 的 啊 certain 的 啦')
'EN'
```

The package includes several implementations of methods for matrix language determination:

1. Word majority from [Bullock et al 2018](https://aclanthology.org/W18-3208/): **MatrixLanguageDeterminerWordMajority**
2. First part of the Morpheme Order Principle from [Myers-Scotton 2002](https://academic.oup.com/book/36360), called the singleton principle in [Iakovenko 2024](https://aclanthology.org/2024.emnlp-main.330/): **MatrixLanguageDeterminerP11**
3. Second part of the the Morpheme Order Principle as in [Iakovenko 2024](https://aclanthology.org/2024.emnlp-main.330/): **MatrixLanguageDeterminerP12**
4. System Morpheme Principle from [Myers-Scotton 2002](https://academic.oup.com/book/36360): **MatrixLanguageDeterminerP2**

### P1.2 matrix language determiner usage

**MatrixLanguageDeterminerP12** requires trained language models for running in order to rescore code-switched sentences. To download the trained models, used in the experiments of [Iakovenko 2024](https://aclanthology.org/2024.emnlp-main.330/), clone the following repository:

```shell
cd /your/model/folder
git clone https://huggingface.co/dinoyay/ml-determination-lms
```

Then you can determine the matrix language using P1.2:

```python
>>> from ml_determination.predict_matrix_language import MatrixLanguageDeterminerP12
>>> config = {
  'EN': {
    'data_path': '/your/model/folder/ml-determination-lms/en/',
    'model_path': '/your/model/folder/ml-determination-lms/en/model.pt'},
  'ZH': {
    'data_path': '/your/model/folder/ml-determination-lms/zh/',
    'model_path': '/your/model/folder/ml-determination-lms/zh/model.pt'
    }
  }
>>> ml = MatrixLanguageDeterminerP12(L1='ZH', L2='EN', config=config, alpha=1.2765)
>>> ml.determine_ML('然后 那些 air supply 的 然后 michael learns to rock 的 啊 certain 的 啦')
'ZH'
```

## Citation
If you use ml_determination in your projects, please feel free to cite the original EMNLP paper the following way:

```
@inproceedings{iakovenko-hain-2024-methods,
    title = "Methods of Automatic Matrix Language Determination for Code-Switched Speech",
    author = "Iakovenko, Olga  and Hain, Thomas",
    editor = "Al-Onaizan, Yaser and Bansal, Mohit and Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.330/",
    doi = "10.18653/v1/2024.emnlp-main.330",
    pages = "5791--5800"
}
```

## Acknowledgements

This code was created with the support of [Engineering and Physical Sciences Research Council](https://gtr.ukri.org/projects?ref=studentship-2676033).
