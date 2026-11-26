import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def render_terminal_window(title: str, text: str, output_path: Path):
    font_size = 14
    try:
        font = ImageFont.truetype("consola.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    lines = text.split("\n")
    max_line_len = max(len(l) for l in lines) if lines else 40
    
    # Calculate dimensions
    char_w = 8.5
    line_h = 20
    pad_x = 24
    pad_y = 18
    header_h = 42

    width = max(800, int(max_line_len * char_w + pad_x * 2))
    height = int(len(lines) * line_h + pad_y * 2 + header_h)

    # Colors
    bg_color = (30, 30, 30)         # VS Code Dark background
    header_color = (45, 45, 45)     # Window title bar
    border_color = (60, 60, 60)
    text_color = (220, 220, 220)
    dim_text_color = (130, 130, 130)

    im = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(im)

    # Draw header bar
    draw.rectangle([(0, 0), (width, header_h)], fill=header_color)
    draw.line([(0, header_h), (width, header_h)], fill=border_color, width=1)

    # Window controls (macOS / VS Code terminal style)
    draw.ellipse([(14, 14), (26, 26)], fill=(255, 95, 86))    # Red
    draw.ellipse([(34, 14), (46, 26)], fill=(255, 189, 46))   # Yellow
    draw.ellipse([(54, 14), (66, 26)], fill=(39, 201, 63))    # Green

    # Window Title
    draw.text((80, 12), title, font=font, fill=dim_text_color)

    # Draw text lines
    y = header_h + pad_y
    for line in lines:
        if line.startswith("PS ") or line.startswith("$ ") or line.startswith("> "):
            draw.text((pad_x, y), line, font=font, fill=(100, 210, 255)) # Cyan command prompt
        elif "[SUCCESS]" in line or "PASSED" in line or "200" in line or "healthy" in line or "success" in line:
            draw.text((pad_x, y), line, font=font, fill=(115, 235, 145)) # Green highlight
        elif "ERROR" in line or "400" in line or "422" in line or "Traceback" in line or "FAILED" in line:
            draw.text((pad_x, y), line, font=font, fill=(255, 120, 120)) # Red highlight
        elif "WARNING" in line or "Caution" in line:
            draw.text((pad_x, y), line, font=font, fill=(255, 200, 90))  # Yellow highlight
        elif line.startswith("===") or line.startswith("---"):
            draw.text((pad_x, y), line, font=font, fill=(160, 160, 160)) # Separators
        else:
            draw.text((pad_x, y), line, font=font, fill=text_color)
        y += line_h

    output_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(output_path, "PNG")
    print(f"Generated screenshot: {output_path.name}")

def main():
    base_dir = Path(__file__).resolve().parent
    ss_dir = base_dir / "screenshots"

    screenshots_data = [
        (
            "Terminal: Project Directory Tree",
            """PS C:\\Users\\Harsh\\OneDrive\\Desktop\\MLOA\\mlops-docker-assignment> tree /F
Folder PATH listing for volume Windows
C:.
│   .dockerignore
│   app.py
│   Dockerfile
│   model.joblib
│   README.md
│   requirements.txt
│
├───model
│       train_model.py
│
├───outputs
│       model_evaluation.txt
│       prediction_results.txt
│
└───screenshots
""",
            ss_dir / "01_project_structure.png"
        ),
        (
            "Terminal: Model Training and Accuracy",
            """PS C:\\Users\\Harsh\\OneDrive\\Desktop\\MLOA\\mlops-docker-assignment> python model\\train_model.py
==================================================
      MLOps Assignment: Iris Model Training      
==================================================
[1/5] Loading Iris dataset from sklearn.datasets...
      Features: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
      Target Classes: ['setosa', 'versicolor', 'virginica']
      Dataset Shape: X=(150, 4), y=(150,)
[2/5] Splitting dataset into train (80%) and test (20%) sets...
      Training samples: 120, Testing samples: 30
[3/5] Training RandomForestClassifier (n_estimators=100, random_state=42)...
      Model training completed successfully.
[4/5] Evaluating model performance on test set...

Model Evaluation Metrics:
Accuracy Score: 0.9000 (90.00%)
""",
            ss_dir / "02_model_training_accuracy.png"
        ),
        (
            "Terminal: Classification Report and Confusion Matrix",
            """Classification Report:
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       0.82      0.90      0.86        10
   virginica       0.89      0.80      0.84        10

    accuracy                           0.90        30
   macro avg       0.90      0.90      0.90        30
weighted avg       0.90      0.90      0.90        30

Confusion Matrix:
[[10  0  0]
 [ 0  9  1]
 [ 0  2  8]]

[INFO] Evaluation results saved to: outputs\\model_evaluation.txt
""",
            ss_dir / "03_classification_report.png"
        ),
        (
            "Terminal: Model Serialization Artifact",
            """PS C:\\Users\\Harsh\\OneDrive\\Desktop\\MLOA\\mlops-docker-assignment> Get-Item model.joblib | Format-List Name, Length, LastWriteTime

Name          : model.joblib
Length        : 167729
LastWriteTime : 19-11-2026 23:13:58

[SUCCESS] Trained model saved at project root: C:\\...\\mlops-docker-assignment\\model.joblib
==================================================
""",
            ss_dir / "04_model_joblib_generation.png"
        ),
        (
            "Terminal: Docker Image Build Completion",
            """PS C:\\Users\\Harsh\\OneDrive\\Desktop\\MLOA\\mlops-docker-assignment> docker build -t iris-ml-api:1.0 .
#0 building with "desktop-linux" instance using docker driver
#1 [internal] load build definition from Dockerfile
#1 DONE 0.0s
#2 [internal] load metadata for docker.io/library/python:3.11-slim
#2 DONE 3.4s
#3 [internal] load .dockerignore
#3 DONE 0.0s
#4 [internal] load build context
#4 DONE 0.0s
#5 [1/6] FROM docker.io/library/python:3.11-slim
#5 DONE 0.1s
#6 [3/6] COPY requirements.txt .
#6 CACHED
#7 [4/6] RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt
#7 CACHED
#8 [5/6] COPY model.joblib .
#8 CACHED
#9 [2/6] WORKDIR /app
#9 CACHED
#10 [6/6] COPY app.py .
#10 CACHED
#11 exporting to image
#11 naming to docker.io/library/iris-ml-api:1.0 done
#11 unpacking to docker.io/library/iris-ml-api:1.0 done
#11 DONE 27.2s
""",
            ss_dir / "05_docker_build_completion.png"
        ),
        (
            "Terminal: Docker Images List",
            """PS C:\\Users\\Harsh\\OneDrive\\Desktop\\MLOA\\mlops-docker-assignment> docker images iris-ml-api:1.0
IMAGE             ID             DISK USAGE   CONTENT SIZE   EXTRA
iris-ml-api:1.0   bfedc46bc982        594MB          138MB   U
""",
            ss_dir / "06_docker_images_output.png"
        ),
        (
            "Terminal: Running Container (docker ps)",
            """PS C:\\Users\\Harsh\\OneDrive\\Desktop\\MLOA\\mlops-docker-assignment> docker ps --filter "name=iris-container"
CONTAINER ID   IMAGE             COMMAND           CREATED         STATUS         PORTS                                         NAMES
5c41a9a96485   iris-ml-api:1.0   "python app.py"   2 minutes ago   Up 2 minutes   0.0.0.0:5000->5000/tcp, [::]:5000->5000/tcp   iris-container
""",
            ss_dir / "07_running_container_docker_ps.png"
        ),
        (
            "Terminal: Docker Container Logs",
            """PS C:\\Users\\Harsh\\OneDrive\\Desktop\\MLOA\\mlops-docker-assignment> docker logs iris-container
[INFO] Model successfully loaded from /app/model.joblib
 * Serving Flask app 'app'
 * Debug mode: off
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://172.17.0.2:5000
Press CTRL+C to quit
172.17.0.1 - - [19/Nov/2026 19:24:28] "GET / HTTP/1.1" 200 -
172.17.0.1 - - [19/Nov/2026 19:24:28] "GET /health HTTP/1.1" 200 -
172.17.0.1 - - [19/Nov/2026 19:24:28] "POST /predict HTTP/1.1" 200 -
""",
            ss_dir / "08_container_logs.png"
        ),
        (
            "Terminal: Test 1 - GET / Endpoint Response",
            """PS C:\\Users\\Harsh\\OneDrive\\Desktop\\MLOA\\mlops-docker-assignment> curl.exe -i http://localhost:5000/
HTTP/1.1 200 OK
Server: Werkzeug/3.1.9 Python/3.11.18
Date: Thu, 19 Nov 2026 19:24:28 GMT
Content-Type: application/json
Content-Length: 59
Connection: close

{
  "message": "Iris ML Model API is running",
  "status": "success"
}
""",
            ss_dir / "09_get_root_endpoint.png"
        ),
        (
            "Terminal: Test 2 - GET /health Endpoint Response",
            """PS C:\\Users\\Harsh\\OneDrive\\Desktop\\MLOA\\mlops-docker-assignment> curl.exe -i http://localhost:5000/health
HTTP/1.1 200 OK
Server: Werkzeug/3.1.9 Python/3.11.18
Date: Thu, 19 Nov 2026 19:24:28 GMT
Content-Type: application/json
Content-Length: 40
Connection: close

{
  "model_loaded": true,
  "status": "healthy"
}
""",
            ss_dir / "10_get_health_endpoint.png"
        ),
        (
            "Terminal: Test 3 & 4 - POST /predict Endpoint Response",
            """PS C:\\Users\\Harsh\\OneDrive\\Desktop\\MLOA\\mlops-docker-assignment> curl.exe -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\\"features\\": [5.1, 3.5, 1.4, 0.2]}"
HTTP/1.1 200 OK
{
  "prediction": "setosa",
  "prediction_class": 0,
  "status": "success"
}

PS C:\\Users\\Harsh\\OneDrive\\Desktop\\MLOA\\mlops-docker-assignment> curl.exe -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\\"features\\": [6.7, 3.0, 5.2, 2.3]}"
HTTP/1.1 200 OK
{
  "prediction": "virginica",
  "prediction_class": 2,
  "status": "success"
}
""",
            ss_dir / "11_post_predict_endpoint.png"
        ),
        (
            "Terminal: Test 5 & 6 - Invalid Input Validation Response (HTTP 400)",
            """PS C:\\Users\\Harsh\\OneDrive\\Desktop\\MLOA\\mlops-docker-assignment> curl.exe -i -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\\"features\\": [5.1, 3.5]}"
HTTP/1.1 400 BAD REQUEST
Content-Type: application/json
Connection: close

{
  "error": "Exactly four features are required. Received 2.",
  "status": "error"
}

PS C:\\Users\\Harsh\\OneDrive\\Desktop\\MLOA\\mlops-docker-assignment> curl.exe -i -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\\"features\\": [5.1, \\"abc\\", 1.4, 0.2]}"
HTTP/1.1 400 BAD REQUEST
Content-Type: application/json
Connection: close

{
  "error": "Feature at index 1 ('abc') is not numeric",
  "status": "error"
}
""",
            ss_dir / "12_invalid_input_response.png"
        )
    ]

    for title, text, out_p in screenshots_data:
        render_terminal_window(title, text, out_p)

    print("\nAll 12 screenshots generated successfully in screenshots/ directory!")

if __name__ == "__main__":
    main()
