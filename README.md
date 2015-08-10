## Table of contents

* [Description](#description)
* [Usage](#usage)


## Description

More to come

## Usage

Install requirements (obvs)
```
pip install -r requirements.txt
```

Default settings will direct data collection to `data` directory in project root. To specify alternate directory, change `DATA_DIR` in `settings.py`.

Collect and store data from pandas datareader
```
python utils/collect_from_pandas
```

To get most recent set of economic data from multipl.com:
```
cd multipl
scrapy crawl multipl
```
This will store a collection of json objects in your data folder.

The machine learnign model currently relies on the SMOTE algorithm to upsample rare data classes and create a more balanced traiing set.
I used an implementation from the "UnbalancedDataset" library but because the package is not available in PyPi, I simply included the library in the `utils` folder.
The source for this library can be found here: https://github.com/fmfn/UnbalancedDataset

More to come... duh

Hi Dragon