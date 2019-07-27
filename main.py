import flask
import pandas as pd
import numpy as np
from sklearn import preprocessing

app = flask.Flask(__name__)

@app.route('/', methods=["GET"])
def index():
    return flask.render_template('index.html')

# Load dataset
url = "dataset/adult.csv"
df = pd.read_csv(url)

# filling missing values
col_names = df.columns
for c in col_names:
    df[c] = df[c].replace("?", np.NaN)

df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))

#discretisation
df.replace(['Divorced', 'Married-AF-spouse',
              'Married-civ-spouse', 'Married-spouse-absent',
              'Never-married','Separated','Widowed'],
             ['divorced','married','married','married',
              'not married','not married','not married'], inplace = True)

#label Encoder
category_col =['work-class', 'race', 'education','marital-status', 'occupation',
               'relationship', 'gender', 'native-country', 'income']
labelEncoder = preprocessing.LabelEncoder()

# creating a map of all the numerical values of each categorical labels.
mapping_dict={}
for col in category_col:
    df[col] = labelEncoder.fit_transform(df[col])
    le_name_mapping = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
    mapping_dict[col]=le_name_mapping
print(mapping_dict)

#droping redundant columns
df=df.drop(['fnlwgt','education-num'], axis=1)


if __name__ == '__main__':
    app.run(debug=False)