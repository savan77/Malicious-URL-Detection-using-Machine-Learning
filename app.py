from flask import Flask, render_template
from flask_wtf import FlaskForm as Form 
from wtforms import StringField
from wtforms.validators import InputRequired, URL
import joblib

import re



app = Flask(__name__)
app.config['SECRET_KEY']= ''

def trim(url):
    return re.match(r'(?:\w*://)?(?:.*\.)?([a-zA-Z-1-9]*\.[a-zA-Z]{1,}).*', url).groups()[0]

def getTokens(input):
    tokensBySlash = str(input.encode('utf-8')).split('/')
    allTokens = []
    for i in tokensBySlash:
        tokens = str(i).split('-')	
        tokensByDot = []
        for j in range(0,len(tokens)):
            tempTokens = str(tokens[j]).split('.')
            tokensByDot = tokensByDot + tempTokens
        allTokens = allTokens + tokens + tokensByDot
    allTokens = list(set(allTokens))
    if 'com' in allTokens:
        allTokens.remove('com')	
    return allTokens


class LoginForm(Form):
	url = StringField('Enter URL : ', validators=[InputRequired(), URL()])


@app.route('/', methods=['GET', 'POST'])
def index():
	form = LoginForm()
	if form.validate_on_submit():
		model = joblib.load('pre-trained/mal-logireg1.pkl')
		vectorizer = joblib.load("pre-trained/vectorizer1.pkl")
		prediction = model.predict(vectorizer.transform([trim(form.url.data)]))

		if prediction[0] == 0:
			#prediction = "NOT MALICIOUS"
			return render_template("success.html", url = form.url.data, status = "Not Malicious")
		else:
			#prediction = "MALICIOUS"
			return render_template("success.html", url= form.url.data, status = "Malicious")
		#return render_template('success.html', url = form.url.data, prediction = prediction)
	return render_template('index.html', form=form)

if __name__ == '__main__':
	app.run(debug=True)
