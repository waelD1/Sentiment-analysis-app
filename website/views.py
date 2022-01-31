from flask import Blueprint, render_template, request, flash, redirect, url_for, session
from .model import predict



views = Blueprint('views', __name__)

@views.route('/', methods = ['GET', 'POST'])
def text_to_summarize():
    if request.method == 'POST':
        text = request.form['text']
        if len(text) < 10:
            flash('The text should be longer in order to analyse it', category = 'error')     
        else:
            session.clear()
            session['text'] = text
            return redirect(url_for('views.result'))
    return render_template('base.html')


@views.route('/result', methods = ['GET', 'POST'])
def result():
    if request.method == 'GET':
        text_to_predict = session.get('text')
        prediction = predict(text_to_predict)
        sentiment = ""
        if prediction >=0.5:
            sentiment = 'Positive Review'
            score = f'{round(prediction*100, 3)} %'
        else:
            sentiment = 'Negative Review'
            score = f'{round((1-prediction)*100, 3)} %'
    if request.method == 'POST':
        text_to_predict = request.form['text']
        if text_to_predict is not None:
            session.clear()
            session['text'] = text_to_predict
            return redirect(url_for('views.result'))
    return  render_template("result.html", text_to_predict=text_to_predict, sentiment=sentiment, score=score)
