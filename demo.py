from flask import Flask
from flask import request
import algo
# Create subprocess with pipes for stdin and stdout


# Reassign the pipes to our stdin and stdout
app = Flask(__name__)

@app.route("/")
def hello():
    return algo.get_prediction(request.args.get('review'))


if __name__ == "__main__":
    app.run()