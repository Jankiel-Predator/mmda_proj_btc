# bitcoin_preprocessed.csv is already preprocessed and you can read it directly

We forgot that **Price**, **Open**, **High**, **Low** have commas indicating thousands, it could not be converted to floats and generated NaN when trying to modify it but I noticed it only after trying to make a chart of the Price movement over time.

## Remember to read it this way:
```
pd.read_csv('data.csv', parse_dates=['Date'])
```

Otherwise the **Date** column may not work correctly. Also, you can set index to **Date** but I do not know whether this is necessary.

---

WORK SMART NOT HARD!!
_____________________________________________________

Clone the repository using:

```
git clone https://github.com/Jankiel-Predator/capstone-bitcoin-gold.git
```

Create new branch, stage, commit, push

Install Plotly!!!


Requirements for the project:
1. Time Series Recognition:
   - Ability to identify time series data.
   - Determination of basic dependencies in time series.
   - Decomposition of time series into components: trend, seasonality, residual series.

2. Examining Time Series Properties:
   - Analysis of stationarity using appropriate statistical tests.
   - Performing transformations on time series.
   - Creating auxiliary plots.

3. Preparation for Modeling:
   - Splitting data into training and test sets.
   - Selecting an appropriate model for time series analysis.
   - Conducting the model training process.
   - Making predictions using the model.
   - Evaluating the effectiveness of the obtained model.


_____________________________________________________
 - To evaluate the project, the contribution of each group member must be clearly stated.
- All sources used in the project must be included. If AI tools were used, a list of prompts should be attached. If AI was not utilized, this must be explicitly noted in the file.
- Every project participant must understand all parts of the project, not only the sections they contributed to. Questions during the final presentation can be directed to any group member.
   - All members of the group receive the same score for the project.

________________________
zobaczyc sobie modele wykladnicze,
zmienic zbior danych na odstep do dzien (closing price) - Wiola,
sprawdzic outliery - Maria,
sprawdzic seasonality/trendy w danych - Maria,
podzielic dataset na podzbiory - Kornelia,
-- po zrobieniu modelu/modeli:
residual analysis with conclusions - ,
statistical tests,
forecasting based on the model and verfication of the forecast
