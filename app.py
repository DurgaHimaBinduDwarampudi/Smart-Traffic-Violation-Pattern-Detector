import os
from flask import Flask, render_template, request
from violation_detector import detect_violations
from plots import generate_chart

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

# Ensure required folders exist
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/results", exist_ok=True)
os.makedirs("static/charts", exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return "No file part"

    file = request.files["image"]
    if file.filename == "":
        return "No selected file"

    if file:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        result_image, violations = detect_violations(file_path)
        chart_path, most_common = generate_chart()

        # Normalize slashes for cross-platform
        result_image = result_image.replace("\\", "/")
        if chart_path:
            chart_path = chart_path.replace("\\", "/")

        # Ensure leading slash for correct browser path
        if not result_image.startswith("/"):
            result_image = "/" + result_image
        if chart_path and not chart_path.startswith("/"):
            chart_path = "/" + chart_path

        return render_template(
            "result.html",
            result_image=result_image,
            violations=violations,
            chart_path=chart_path,
            most_common=most_common
        )

if __name__ == "__main__":
    app.run(debug=True)
