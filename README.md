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
