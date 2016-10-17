from flask import Flask,url_for,render_template,request
from tfwrapper import imageRecognizer as ir
app = Flask(__name__,static_url_path='')

@app.route("/")
# @app.route("/index.html")
def index():
    return render_template('index.html')

@app.route('/upload',methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f=request.files['image']
        print('====Image type:',type(f))
        binaryFile = f.read()
        results = ir.inferenceImage(binaryFile)
        
        return render_template('result.html',results = results)

@app.route('/post/<int:post_id>')
def show_post(post_id):
    # show the post with the given id, the id is an integer
    return 'Post %d' % post_id


# with app.test_request_context():
#   print ('====================',url_for('index'))
#   url_for('static', filename='style.css')

if __name__ == "__main__":
    app.run()
