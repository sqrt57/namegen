# Character-level language model for generating names

## Plan
1. Get several names datasets
2. Simple bigrams model
3. Simple one-layer NN equivalent to birgrams model
4. MLP

## Data
### UK towns and counties names
First data (data/raw/uk-towns.txt) was scraped from website:
https://www.britinfo.net/

Better data (data/raw/uk-towns.csv), more than 10 times bigger was scraped from:
https://www.townscountiespostcodes.co.uk/towns-in-uk/

### USA names list
https://www.ssa.gov/oact/babynames/limits.html

### Russian names and surnames
https://www.kaggle.com/datasets/rai220/russian-cyrillic-names-and-sex


## References
* [Andrej Karpathy: A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)
* [Cookiecutter Data Science: A logical, flexible, and reasonably standardized project structure for doing and sharing data science work.](https://cookiecutter-data-science.drivendata.org/)
* [Andrej Karpathy: makemore](https://github.com/karpathy/makemore)
* [Yoshua Bengio, Réjean Ducharme, Pascal Vincent, Christian Jauvin 2003: A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)