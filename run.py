
from plant_detection import create_app
from plant_detection.routes import bp

app = create_app()
app.register_blueprint(bp)

if __name__ == '__main__':
    app.socketio.run(app, host='0.0.0.0', port=5002, debug=False)
