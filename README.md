# myphdthesis

Structure taken from [@maxnoe](https://github.com/maxnoe/TuDoThesis) with small design adaptations.

## Requirements

[TexLive 2020](https://tug.org/mactex/) is used to compile the document.

For the plotting scripts python 3.8 with the following packages is used:

- numpy==1.19.5
- scipy==1.6.0
- matplotlib==3.3.3
- proposal==7.0.2
- python-ternary==1.0.7

## Further Software used

Python libraries that are not necessary to run the scripts, but used during development

- tqdm==4.56.0
- jupyter==1.0.0

To crop/extract images from external resources (paper or proceeding) the following software was used:

- [Inkscape](https://gitlab.com/inkscape/inkscape) version 1.0.2
- pdfcrop version 1.38 (included in TexLive)
- [FORM](https://github.com/vermaseren/form) version 4.2.1

## Reproducibility

to run the scripts on vollmond, load the required software packages
```
module add python/3.9.4
scl enable devtoolset-9 bash
```
and create a virtual env
```
mkdir ~/venvs
/opt/python/3.9.4/bin/python -m venv ~/venvs/vthesis
source ~/venvs/vthesis/bin/activate
pip install --upgrade pip
pip install numpy, scipy, matplotlib, proposal
```