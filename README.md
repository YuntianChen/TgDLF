# TgDLF
Theory-guided deep-learning load forecasting is a short-term load forecasting model that combines domain knowledge and machine learning algorithms. 

The model can predict future load based on historical load, weather forecast data, and the calendar effect. Specifically, the grid load is first converted into a load ratio to avoid the impact of different data distributions in different time periods in the same district. The load ratio is then decomposed into dimensionless trends that can be calculated in advance based on domain knowledge, and local fluctuations that are estimated by EnLSTM (please refer to the project of EnLSTM for more details: https://github.com/YuntianChen/EnLSTM). 
