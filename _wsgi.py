from label_studio_ml.api import init_app
from yolo import MyModel

# for uWSGI use
app = init_app(model_class=MyModel)

if __name__ == "__main__":
    # For direct Python execution (development/testing)
    app.run(host='0.0.0.0', port=9090, debug=False)
