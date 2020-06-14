# covid19
Covid-19 jupyter notebook that calculates covid19 trends via regression and produces some neat output graphs.

## How to install and run

Install via
```bash
git clone https://github.com/jvm123/covid19
cd covid19
virtualenv .
source bin/activate
python3 -m pip install ipywidgets numpy pandas matplotlib sklearn plotly
```

To run the command line version that exports graphs based on the newest [Johns Hopkins CSSE data](https://github.com/CSSEGISandData/COVID-19):
```bash
python3 covid19_live.py
```

To open the jupyter notebook, run:
```bash
jupyter notebook
```
and select the covid19_live.ipynb file.

After modifying the jupyter notebook, you can derive an updated command line version via
```bash
jupyter nbconvert --to script --execute covid19_live.ipynb
```
The python program covid19_live.py is the result.

