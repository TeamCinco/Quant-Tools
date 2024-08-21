from flask import Flask, request, render_template_string
import pandas as pd
from io import StringIO

app = Flask(__name__)

template = '''
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Option Data Upload</title>
  </head>
  <body>
    <h1>Upload Calls and Puts Data</h1>
    <form action="/" method="post">
      <label for="calls">Calls Data:</label><br>
      <textarea id="calls" name="calls" rows="10" cols="100"></textarea><br><br>
      <label for="puts">Puts Data:</label><br>
      <textarea id="puts" name="puts" rows="10" cols="100"></textarea><br><br>
      <input type="submit" value="Submit">
    </form>
    {% if message %}
      <p>{{ message }}</p>
    {% endif %}
  </body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    message = ''
    if request.method == 'POST':
        calls_data = request.form['calls']
        puts_data = request.form['puts']

        if calls_data:
            calls_df = parse_option_data(calls_data)
            calls_df.to_csv('calls.csv', index=False)
            message += 'Calls data saved to calls.csv. '

        if puts_data:
            puts_df = parse_option_data(puts_data)
            puts_df.to_csv('puts.csv', index=False)
            message += 'Puts data saved to puts.csv.'

    return render_template_string(template, message=message)

def parse_option_data(data):
    # Replace \t with commas to ensure proper CSV formatting
    data = data.replace('\t', ',')
    df = pd.read_csv(StringIO(data), sep=',')
    return df

if __name__ == '__main__':
    app.run(debug=True)
