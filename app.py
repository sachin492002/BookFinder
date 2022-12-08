from flask import Flask, render_template, request
from indexer import *

app = Flask(__name__, template_folder='templates', static_folder='static')


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # book = request.args.get('book')
        book = dict(request.form)
        books = searchFlask(book['search'])
        # print(books)
    else:
        books = []

    return render_template("index.html", books=books)


# @app.route("/search")
# def search():
#     if request.method == "GET":
#         book = dict(request.form)
#         books = searchFlask(book['search'])
#     return render_template("index.html", books=books)


if __name__ == '__main__':
    app.run(debug=True)
