# Textual supervisoin for visually grounded spoken language understanding

This repository contains the instructions and code to reproduce the results from the paper:

> Higy, B., Elliott, D. & Chrupała, G. Textual supervision for visually grounded spoken language understanding. In Findings of Empirical Methods in Natural Language Processing. Punta Cana, Dominican Republic, 2020

## Prerequisites

- Download the [Flickr8K dataset](https://forms.illinois.edu/sec/1713398) as well as the [Flickr Audio Caption Corpus](https://groups.csail.mit.edu/sls/downloads/flickraudio/). The audio captions should be extracted under the same folder as the rest of the dataset.
- Update the parameter `flickr8k_root` in the file `config.yml` to point to the location of the dataset. Make `~/.platalea/config.yml` point to this file by running:

        mkdir ~/.platalea
        ln -s config.yml ~/.platalea/config.yml

- Copy the files `dataset.json` and `dataset_multilingual_human.json` to the root folder of the Flickr8K dataset.
- Install the requirements:
```
pip install -r requirements.txt
```

## Training the models

The different models presented in the paper can be trained by running the script `run.sh`:

    ./run.sh

## Reproducing the figures and tables from the paper

Figure 3 can be reproduced by running:

    python -c "import results; results.plot_figure_3()"

Results presented in Tables 1-3 and 6-10 can be extracted by running:

    python -c "import results; results.print_table_1()"
    python -c "import results; results.print_table_2()"
    python -c "import results; results.print_table_3()"
    python -c "import results; results.print_table_6()"
    python -c "import results; results.print_table_7()"
    python -c "import results; results.print_table_8()"
    python -c "import results; results.print_table_9()"
    python -c "import results; results.print_table_10()"

## Credits

The files `dataset.json` and `dataset_multilingual_human.json` are based on pre-processed information extracted from Flickr8K, Flickr8K Audio Caption Corpus and F30kEnt-JP. While they are provided here for conveniency, credits go to the authors of the original datasets.

* Flickr8K:
> Hodosh, M., Young, P., & Hockenmaier, J. (2013). Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics. Journal of Artificial Intelligence Research, 47, 853–899. https://doi.org/10.1613/jair.3994
* Flickr8K Audio Caption Corpus:
> Harwath, D., & Glass, J. (2015). Deep multimodal semantic embeddings for speech and images. 2015 IEEE Workshop on Automatic Speech Recognition and Understanding (ASRU), 237–244. https://doi.org/10.1109/ASRU.2015.7404800
* F30kEnt-JP:
> Nakayama, H., Tamura, A., & Ninomiya, T. (2020). A Visually-Grounded Parallel Corpus with Phrase-to-Region Linking. Proceedings of The 12th Language Resources and Evaluation Conference, 4197–4203. https://www.aclweb.org/anthology/2020.lrec-1.518

