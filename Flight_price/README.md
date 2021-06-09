
# Business problem

Travel agencies need to track and predict fluctuations in flight prices to provide competitive tour packages. In this project we try to build a model to estimate flight prices for direct flights in five routes. This can be expanded to include more routes after more data is scraped from travel websites.


# Data understanding

## Available features

`Airline` : The flight carrier

`Date_of_Journey`: The date the flight flies

`Source`: Travel origin

`Destination`: Travel destination

`Route`: Whether the flight was hopping (Not used in modelling)

`Dep_Time`: Departure time at origin

`Arrival_Time`: Arrival time at destination (Not used in modelling)

`Duration`: Flight duration

`Total_Stops`: No. of stops (Not used in modelling)

`Additional_Info`: Miscellaneous information (Not used in modelling)

`Price`: Target continuous variable

## Additional features generated(other than One-hot encoding etc.)

`Route`: Unique identification of route

`TOD of departure`: Part of the day when the flight originated from source

# Model preparation

After trying out different models and doing hyper parameter tuning, Random Forest performed the best. 

# Model evaluation

I used MAE as the evaluation metric. This was done so that the travel company's stakeholders could easily interpret the result and factor in this error in their travel package cost calculation.

Absolute error for Decision tree regressor: Rs. 288.65 ± 25.49

Absolute error for Random forest regressor: Rs. 263.85 ± 15.06

Absolute error for XGB Regressor: Rs. 322.58 ± 14.56


# Model deployment

The model was deployed using Streamlit. The following details was captured:
* Date of planned flight
* Time of planned flight
* Route selection ( I plan on adding source and destination once both way flight data is available) 
* Prefered choice of airline

Depending on the following input the model outputs the price. On some days, certain carrier's flights are not available. Try to spot these in the app! Here you [go:](https://share.streamlit.io/coderkol95/data-science-projects/flight_app.py)
