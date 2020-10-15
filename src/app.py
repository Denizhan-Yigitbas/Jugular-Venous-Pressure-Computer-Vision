import os

from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image 

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
target = os.path.join(APP_ROOT, 'images/')
if not os.path.isdir(target):
    os.mkdir(target)

UPLOAD_FOLDER = target
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# check if selected file is verified
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            # TODO: What is this doing?
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            # TODO: Think about this?
            # flash('No selected file')
            # return redirect(request.url)
            return render_template('upload.html', no_file_selected=True)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file', filename=filename))
    return render_template('upload.html', no_file_selected=False)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    image_file = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename)) # open colour image
    image_file = image_file.convert('1') # convert image to black and white
    image_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")