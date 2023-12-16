# AI-Tumor-Code
# The purpose of the code below is to determine whether or not the tumors are malignant(cancerous) or benign(not cancerous)
# Here is the imported data
diagnosis(1=malignant, 0=benign)	radius_mean	texture_mean	perimeter_mean	area_mean	smoothness_mean	compactness_mean	concavity_mean	concave points_mean	symmetry_mean	fractal_dimension_mean	radius_se	texture_se	perimeter_se	area_se	smoothness_se	compactness_se	concavity_se	concave points_se	symmetry_se	fractal_dimension_se	radius_worst	texture_worst	perimeter_worst	area_worst	smoothness_worst	compactness_worst	concavity_worst	concave points_worst	symmetry_worst	fractal_dimension_worst
0	13.54	14.36	87.46	566.3	0.09779	0.08129	0.06664	0.04781	0.1885	0.05766	0.2699	0.7886	2.058	23.56	0.008462	0.0146	0.02387	0.01315	0.0198	0.0023	15.11	19.26	99.7	711.2	0.144	0.1773	0.239	0.1288	0.2977	0.07259
0	13.08	15.71	85.63	520	0.1075	0.127	0.04568	0.0311	0.1967	0.06811	0.1852	0.7477	1.383	14.67	0.004097	0.01898	0.01698	0.00649	0.01678	0.002425	14.5	20.49	96.09	630.5	0.1312	0.2776	0.189	0.07283	0.3184	0.08183
0	9.504	12.44	60.34	273.9	0.1024	0.06492	0.02956	0.02076	0.1815	0.06905	0.2773	0.9768	1.909	15.7	0.009606	0.01432	0.01985	0.01421	0.02027	0.002968	10.23	15.66	65.13	314.9	0.1324	0.1148	0.08867	0.06227	0.245	0.07773
0	13.03	18.42	82.61	523.8	0.08983	0.03766	0.02562	0.02923	0.1467	0.05863	0.1839	2.342	1.17	14.16	0.004352	0.004899	0.01343	0.01164	0.02671	0.001777	13.3	22.81	84.46	545.9	0.09701	0.04619	0.04833	0.05013	0.1987	0.06169
0	8.196	16.84	51.71	201.9	0.086	0.05943	0.01588	0.005917	0.1769	0.06503	0.1563	0.9567	1.094	8.205	0.008968	0.01646	0.01588	0.005917	0.02574	0.002582	8.964	21.96	57.26	242.2	0.1297	0.1357	0.0688	0.02564	0.3105	0.07409
0	12.05	14.63	78.04	449.3	0.1031	0.09092	0.06592	0.02749	0.1675	0.06043	0.2636	0.7294	1.848	19.87	0.005488	0.01427	0.02322	0.00566	0.01428	0.002422	13.76	20.7	89.88	582.6	0.1494	0.2156	0.305	0.06548	0.2747	0.08301
0	13.49	22.3	86.91	561	0.08752	0.07698	0.04751	0.03384	0.1809	0.05718	0.2338	1.353	1.735	20.2	0.004455	0.01382	0.02095	0.01184	0.01641	0.001956	15.15	31.82	99	698.8	0.1162	0.1711	0.2282	0.1282	0.2871	0.06917
0	11.76	21.6	74.72	427.9	0.08637	0.04966	0.01657	0.01115	0.1495	0.05888	0.4062	1.21	2.635	28.47	0.005857	0.009758	0.01168	0.007445	0.02406	0.001769	12.98	25.72	82.98	516.5	0.1085	0.08615	0.05523	0.03715	0.2433	0.06563
0	13.64	16.34	87.21	571.8	0.07685	0.06059	0.01857	0.01723	0.1353	0.05953	0.1872	0.9234	1.449	14.55	0.004477	0.01177	0.01079	0.007956	0.01325	0.002551	14.67	23.19	96.08	656.7	0.1089	0.1582	0.105	0.08586	0.2346	0.08025
0	11.94	18.24	75.71	437.6	0.08261	0.04751	0.01972	0.01349	0.1868	0.0611	0.2273	0.6329	1.52	17.47	0.00721	0.00838	0.01311	0.008	0.01996	0.002635	13.1	21.33	83.67	527.2	0.1144	0.08906	0.09203	0.06296	0.2785	0.07408

# Imports the dataset above
import pandas as pd
dataset = pd.read_csv('cancer.csv')

# Sets x equals everything except the diagnosis 
x = dataset.drop(columns=["diagnosis(1=m, 0=b)"])

# Sets y equals to the diagnosis 
y = dataset["diagnosis(1=m, 0=b)"]

# Sets the data set into a testing set and a training set to minigate overfitting(The AI does well with given data but not too well with receiving new data)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Using Keras(Building a neuro network) 
import tensorflow as tf
model = tf.keras.models.Sequential()

# Adding layers to the module(referencing neural network)
# Output would be 256 neurons
# Signmoid is used to set all values within it(reduces model complexity and makes the model more accurate)

model.add(tf.keras.layers.Dense(256, input_shape=x_train.shape[1:], activation='sigmoid'))
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Using the optimizer adam(finetunes the neurons and weights of the algorithm to fit the data)
# Using binary classsification so I used binary_crossentropy 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=('accuracy'))

# Fits the x_train to the y_train 
# Sets epics to 1000 which makes the algorithm run the data that number of times
model.fit(x_train, y_train, epochs=1000)

# Here is the data but ran 1000 times becomming more accurate: 
Epoch 1/1000 15/15 [==============================] - 1s 3ms/step - loss: 0.5908 - accuracy: 0.6901

# Comparing what the model thinks the model tests thinks it should be to what it actually is 
# This determines the level of accuracy the AI can determine whethSer or not the tumor is cancerous or not
# Achieved a 96 percent in determinationm
model.evaluate(x_test, y_test)
4/4 [==============================] - 0s 2ms/step - loss: 0.0857 - accuracy: 0.9649
[0.08568435907363892, 0.9649122953414917]
