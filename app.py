from lncrnanet2 import predict

from flask import Flask, request

import logging

app = Flask(__name__)

logging.basicConfig(filename='logs/flask.log', level=logging.DEBUG, format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')


@app.route('/predict_lncrnanet2', methods=['POST'])
def predict_lncrnanet2():
    try:
        seq = request.form['seq']
        app.logger.info('Start sequence: ' + str(seq))
        if len(seq) < 200:
            raise ValueError('<h1>Sequence length less than 200</h1>')
        elif len(seq) > 3000:
            raise ValueError('<h1>Sequence length more than 3000<h1>')
        else:
            pred = predict.predict_lncrnanet2(seq)
            return {'pred': str(pred)}
    except ValueError as valueError:
        app.logger.error(str(valueError))
        return str(valueError)
    except Exception as exception:
        app.logger.error(str(exception))
        return str(exception)


if __name__ == "__main__":
    app.run(debug=True)
