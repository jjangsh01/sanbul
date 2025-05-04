import tensorflow as tf
from tensorflow import keras

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)

import numpy as np
import pandas as pd

from flask import Flask, render_template
from flask_bootstrap import Bootstrap5
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

STRING_FIELD = StringField('max_wind_speed', validators=[DataRequired()])

np.random.seed(42)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
bootstrap5 = Bootstrap5(app)

class LabForm(FlaskForm):
    longitude = StringField('longitude(1-7)', validators=[DataRequired()])
    latitude = StringField('latitude(1-7)', validators=[DataRequired()])
    month = StringField('month(01-Jan ~ Dec-12)', validators=[DataRequired()])
    day = StringField('day(00-sun ~ 06-sat, 07-hol)', validators=[DataRequired()])
    avg_temp = StringField('avg_temp', validators=[DataRequired()])
    max_temp = StringField('max_temp', validators=[DataRequired()])
    max_wind_speed = StringField('max_wind_speed', validators=[DataRequired()])
    avg_wind = StringField('avg_wind', validators=[DataRequired()])
    submit = SubmitField('Submit')

###############################
    

# 1-1 Data 불러오기
fires = pd.read_csv("./sanbul2district-divby100.csv", sep=",")
fires["burned_area"] = np.log(fires["burned_area"] + 1)

train_set, test_set = train_test_split(fires, test_size=0.2, random_state=42)
test_set.head()

fires["month"].hist()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(fires, fires["month"]):
    strat_train_set = fires.loc[train_index]
    strat_test_set = fires.loc[test_index]

fires = strat_train_set.drop(["burned_area"], axis=1) # drop labels for training set
fires_labels = strat_train_set["burned_area"].copy()

fires_num = fires.drop(["month", "day"], axis=1)

num_attribs = ['longitude', 'latitude',
               'avg_temp', 'max_temp',
               'max_wind_speed', 'avg_wind']
cat_attribs = ['month', 'day']

num_pipeline = Pipeline([
('std_scaler', StandardScaler()),
])

full_pipeline = ColumnTransformer([
("num", num_pipeline, num_attribs),
("cat", OneHotEncoder(), cat_attribs),
])

full_pipeline.fit(fires)  

#######################################


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def lab():
    form = LabForm()
    if form.validate_on_submit():

        X_test = {
        'longitude':       form.longitude.data,
        'latitude':        form.latitude.data,
        'avg_temp':        form.avg_temp.data,
        'max_temp':        form.max_temp.data,
        'max_wind_speed':  form.max_wind_speed.data,
        'avg_wind':        form.avg_wind.data,
        'month':           form.month.data,
        'day':             form.day.data,
        }

        X_test_df = pd.DataFrame([X_test])
        print(X_test_df)

        X_prepared = full_pipeline.transform(X_test_df)

        if hasattr(X_prepared, "toarray"):
            X_prepared = X_prepared.toarray()
        print("▶ 전처리된 배열:\n", X_prepared)

        model = keras.models.load_model('fires_model.keras')

        # evaluate model
        prediction = model.predict(X_prepared)
        



        print(prediction)

        res = prediction[0][0]
        print(res)

        burned_area = np.exp(res) - 1
        print(burned_area)

        res = round(burned_area, 2)
        
        

 
        return render_template('result.html', res=res)

    return render_template('prediction.html', form=form)

