# Data Science Project Report on Automobile Price Prediction
## BUSINESS CASE: BASED ON THE GIVEN FEATURE OF DATASET WE NEED TO PREDICT THE AUTOMOBILE PRICE
### Abstract:
This project aims to develop a robust automobile price prediction model using machine learning techniques. The primary objective is to create a tool that can accurately estimate the market value of automobiles based on various features and specifications.
Methodology involves data preprocessing, exploratory data analysis (EDA), feature engineering, and the application of machine learning algorithms. We used  regression models, including linear regression, decision trees, to build and evaluate our price prediction models.

### Device Project In-to Multiple Steps:
1.	Data Collection 
1.	2.Loading data
2.	3.Domain Analysis 
3.	4.Basic Checks of data
4.	EDA (Univariate, Bivariate, Multivariate Analysis)
5.	Data Pre-processing 
6.	Feature Selection 
7.	Building ML Model
8.	Training & Model Evaluation
9.	Model Savings

### Loading data:
load data in python using pandas library


### Domain Analysis
#### 1. Symboling:
- Symboling is a numerical representation of the risk associated with a vehicle. It typically ranges from -3 to +3, where negative values indicate a lower risk, and positive values indicate a higher risk. It can be used by insurance companies to determine insurance premiums.

#### 2. Normalized-Losses:
Normalized-Losses is a feature in automobile dataset that represents normalized insurance losses in monetary terms. This feature provides insights into the average loss payment per insured vehicle year, normalized for different car models.

#### 3. Make:
The Make feature in automobile dataset refers to the manufacturer or brand of the vehicle. It's a categorical feature that provides information about the company that produced the car.
The "Make" feature represents the name of the car manufacturer or brand, such as "Toyota", "Honda","BMW" and so on.

#### 4. Fuel-Type:
The Fuel Type feature in a dataset refers to the type of fuel used to power vehicles. It's a categorical attribute that classifies vehicles based on the energy source they rely on for their engines.
The two main categories are usually "Diesel" and "Gas" fuel.
Diesel Fuel:
Diesel engines run on diesel fuel, which is less refined than gas. Diesel vehicles tend to be more fuel-efficient and have better torque, making them suitable for heavy-duty applications.
Gas Fuel:
Gas engines use gas as their primary fuel source. Gas is more commonly used for everyday vehicles and has different combustion characteristics compared to diesel.

#### 5. Aspiration:
The Aspiration feature in a dataset for automobiles refers to the type of air intake system used in the vehicle's engine. It's a categorical attribute that indicates whether the engine is naturally aspirated or equipped with a turbocharger for forced induction.
"Aspiration" classifies engines into two main categories: "Standard" (often denoted as "std") and "Turbocharged" (often denoted as "turbo").
Standard (Naturally Aspirated) Engines:
Engines with standard aspiration rely on the natural atmospheric pressure to draw in air during the intake stroke.
Turbocharged Engines:
Turbocharged engines have a turbocharger, which is a device that forces additional air into the engine's cylinders.

#### 6. Num-of-Doors:
The Num-of-Doors feature in a dataset refers to the number of doors a vehicle has. It's a categorical attribute that provides information about the physical design of the vehicle's cabin.
The "Num-of-Doors" feature indicates how many doors a vehicle has. It's a categorical variable that typically has two possible values: "Two" and "Four."

#### 7. Body-Style:
The Body Style feature in an automobile dataset refers to the specific design or configuration of the vehicle's exterior and structure. It's a categorical attribute that categorizes vehicles based on their overall shape and appearance.
Represents the body style of the car (sedan, hatchback, etc.).

#### 8. Drive-Wheels:
The Drive Wheels feature in an automotive dataset refers to the configuration of wheels that are responsible for propelling the vehicle. It's a categorical attribute that provides information about how power is distributed to the wheels.
The main categories are
1."4WD" (Four-Wheel Drive)- Four-wheel drive distributes power to all four wheels, which can improve traction on various terrains.
2."FWD" (Front-Wheel Drive)- In a front-wheel drive configuration, the engine's power is transmitted to the front wheels.
3."RWD" (Rear-Wheel Drive)- In a rear-wheel drive configuration, the engine's power is sent to the rear wheels.

#### 9. Engine-Location:
The Engine Location feature in a dataset refers to the position where the engine of a vehicle is located within the vehicle's chassis. It's a categorical attribute that provides information about the arrangement of the engine.
Front Engine:
The most common engine location is at the front of the vehicle. In front-engine vehicles, the engine is positioned in the front compartment, usually under the hood.
Rear Engine:
In rear-engine vehicles, the engine is located at the rear of the vehicle, typically behind the rear axle.

#### 10. Wheel-Base:
Wheel Base is a key feature in your automobile dataset that refers to the distance between the centers of the front and rear wheels of a vehicle. This measurement provides insights into the vehicle's stability, handling, and overall dimensions.

#### 11. Length:
the Length feature represents the length of a vehicle. It's a numerical attribute that provides information about the physical dimensions of the car.

#### 12. Width:
The Width feature in your automobile dataset refers to the width of the vehicle. It's a numerical attribute that provides information about the horizontal dimensions of a car.

#### 13. Height:
The Height feature in your automobile dataset refers to the vertical measurement of the vehicle from the ground to its highest point.

#### 14. Curb-Weight:
Curb Weight is an important feature in your automobile dataset that refers to the weight of a vehicle when it's fully equipped with all standard items, including essential fluids, fuel, and a full complement of passengers and cargo.

#### 15. Engine-Type:
The Engine Type feature in your automobile dataset describes the type of engine used in each vehicle. It's a categorical attribute that provides information about the configuration and design of the internal combustion engine.
Represents the type of engine (dohc, ohc, etc.).

#### 16. Num-Of-Cylinders:
Indicates the number of cylinders in the engine.
The Num-of-Cylinders feature in your automobile dataset indicates the number of cylinders in the engine of each vehicle. It's a categorical feature that provides information about the internal combustion configuration of the engine.

#### 17. Engine-Size:
Represents the size of the engine in cubic centimeters.
The Engine Size feature in your automobile dataset refers to the displacement volume of an engine. It's a numerical attribute that provides information about the capacity of the engine's combustion chambers.

#### 18. Fuel-System:
Represents the type of fuel system.
The Fuel-System feature in your automobile dataset provides information about the type of fuel delivery system used in the vehicle's engine. It's a categorical feature that offers insights into how fuel is mixed and delivered to the engine for combustion.

#### 20. Bore:
Bore is a feature commonly found in automobile datasets that represents the diameter of each cylinders in an engine. It's a crucial parameter that directly influences the engine's performance, efficiency, and overall characteristics.

#### 20.Stroke:
Stroke is a feature that appears to be related to engine specifications in your automobile dataset. It represents the length of the piston stroke within an engine.

#### 21.Compression-Ratio:
Stroke is the distance that the piston travels inside the cylinder during the engine's operation. It affects engine displacement and performance.

#### 22. Horsepower:
Horsepower is a term used to measure the engine's power output, particularly in vehicles like cars, motorcycles, and other machinery. It's a critical metric that indicates the engine's ability to perform work over time.

#### 24. Peak-rpm:
Peak RPM is a feature in automobile dataset that represents the engine speed at which the engine delivers its maximum power output. It's an important metric for understanding the performance characteristics of a vehicle's engine.

#### 24. City-mpg:
City-mpg, short for "city miles per gallon," is a feature in automobile dataset that provides information about a vehicle's fuel efficiency when driven in city conditions. It measures how many miles a car can travel on average per gallon of fuel consumed while driving in urban or city settings.

#### 26. Highway-mpg:
Highway-mpg is a feature in automobile dataset that represents a measure of fuel efficiency for a vehicle when driving on the highway. This feature provides insights into how many miles a car can travel per gallon of fuel consumed on highways.

#### 26. Price:
Price is a fundamental and critical feature in automobile dataset that represents the cost of a vehicle. This feature provides valuable insights into the financial aspect of different car models and is a key factor influencing purchasing decisions.
Price is Target variable, typically in a specified currency (e.g., USD).

### Exploratory Data Analysis (Univariate, Bivariate, Multivariate Analysis)
#### 1.	Univariate Data Analysis
Use sweetviz library and generate a html report of all feature to do univariate analysis, In that we get the Minimum, Maximum, Some statistical information of the particular feature.
#### 2.	Bivariate Data Analysis
In Bivariate analysis we check the relation of independent feature with respect to target variable

#### 3.	Multivariate Data Analysis¶
In Multivariate analysis we check the relation of two independent feature with respect to target variable

### Data Preprocessing
•	First we check the missing values, and we seen that the  5-6 feature has contain missing value and impute them with median and mode.
•	Handle categorical data and use Manual encoding and frequency encoding. Because features has contain lots of label.
•	In this data I’m Clearly seen that some feature has lots of outlier & we impute them, for that first we check the distribution of all feature and plot the box plot and decide the technique.
•	Scale the numerical independent feature with the help of standard scalar and scale the feature. Because standard scaling give me the best result of ML model.

### Feature Scaling
•	Check the correlation with the help of heatmap and seen the two feature has highly correlated with each other, and we decide to drop one feature i.e City-mpg
•	The dataset not contain any duplicates

### Model Creation & Evaluation
•	Define Independent and dependant variable and split the data into training and testing.
•	For Training 80% & Testing 20% data
•	Use linear regression ,decision tree XGBoost regression algorithm to get a best result and XGB regression is give best result. i.e R2 Score 92.28% and Adjusted R2 Score is 80.72% with 2424 RMSE 

### Model Saving
·	Save the model using pickle file 




