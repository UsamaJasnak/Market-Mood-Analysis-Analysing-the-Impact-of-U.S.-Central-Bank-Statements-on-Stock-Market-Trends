# Market Mood Analysis: Analyzing the Impact of US Central Bank Statements on Stock Market Trends

## Contributors
* [Mohammed Usama Jasnak](https://github.com/UsamaJasnak/)
* [Konstantin Zuev](https://github.com/ghginm)
* [Pallavi Singh](https://github.com/psingh510)
* [Rutvik Vengurlekar](rt762740@dal.ca@dal.ca@dal.ca)


### Project structure

```
NLP/
│
├── data/
│   ├──  output_speech_us_central_bank_v2.parquet   #Labelled input data
│   ├──  speech_us_central_bank.parquet             #Raw and unlabelled input data
│   └──  us_macroeconomic_indicators.parquet        #Macroeconomic variable
├── model/
│   ├── lstm/       #LSTM Mode 
│   ├── gru/         #GRU model
│   └── rf/         #Random forest model
├── scripts/
│   ├── nlp_webscrapping.ipynb  #Scripts for webscrapping data
│   ├── lstm_model.py           #Scripts for training LSTM model
│   ├── gru_model.py           #Scripts for training GRU model
│   ├── preprocessing.py        #Scripts for preprocessing data 
│   └── utils.py                #Scripts for preparing classification report
├── r_scripts/                  #Regression models for analysing stock markets
├── eda.ipynb                   # Jupyter Notebook containing initial EDA of the data
├── final_modelling.ipynb       # The main code for training different models
└── eda_final.ipynb             # Jupyter Notebook containing stock market and correlation analysis wrt to labels
```

