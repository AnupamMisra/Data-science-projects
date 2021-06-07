
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

`Total_Stops`: No. of stops

`Additional_Info`: Miscellaneous information

`Price`: Target continuous variable

## Additional features generated(other than One-hot encoding etc.)

`Route`: Unique identification of route

`TOD of departure`: Part of the day when the flight originated from source

# Model preparation and evaluation

After trying out different models and doing hyper parameter tuning, Random Forest performed the best. I used MAE as the evaluation metric. This was done so that the travel company's stakeholders could easily interpret the result and factor in this error in their travel package cost calculation.

Absolute error for Decision tree regressor: Rs. 292.6 ± 26.6

Absolute error for Random forest regressor: Rs. 264.36 ± 17.33

Absolute error for XGB Regressor: Rs. 322.58 ± 14.56

Some of the important features as per Random Forest in reducing order of importance is as below:
1. Day of year
2. Duration of flight (hours)
3. Duration of flight (mins)
4. Route
5. Whether the flight is Indigo
6. Whether in flight meal was included
7. Departure time(mins)
8. Whether the flight is SpiceJet
9. Departure time(hours)

It is good to have our hunches confirmed by data.
