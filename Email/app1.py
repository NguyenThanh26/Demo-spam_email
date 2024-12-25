from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Tải mô hình đã lưu và vectorizer
with open('svm_model_1.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer_1.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        email = request.form['email']
        # Tiền xử lý văn bản
        data = [email]
        vect = vectorizer.transform(data).toarray()
        prediction = model.predict(vect)
        
        # Trả về kết quả phân loại
        if prediction == 1:
            return render_template('result.html', prediction="Spam")
        else:
            return render_template('result.html', prediction="Ham (Thư hợp lệ)")

if __name__ == "__main__":
    app.run(debug=True)
