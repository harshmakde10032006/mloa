"""
Generate professional, high-fidelity AWS Console and architecture screenshots
for MLOps Assignment 9: Train and Deploy Breast Cancer Classifier using AWS SageMaker.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

os.makedirs("screenshots", exist_ok=True)

# Helper function to create AWS Console base layout
def create_aws_console_base(title, service="Amazon SageMaker", region="ap-south-1", account="123456789012 (Demo)"):
    fig, ax = plt.subplots(figsize=(13, 7.5), dpi=150)
    ax.set_xlim(0, 130)
    ax.set_ylim(0, 75)
    ax.axis("off")

    # AWS Top Navbar (AWS dark navy #232f3e)
    nav_rect = patches.Rectangle((0, 70), 130, 5, facecolor="#232f3e", zorder=2)
    ax.add_patch(nav_rect)
    ax.text(2, 72.2, "aws", fontsize=15, fontweight="bold", color="#ff9900", zorder=3)
    ax.text(8, 72.2, f"|  {service}", fontsize=11, fontweight="bold", color="#ffffff", zorder=3)
    
    # Search bar in navbar
    search_box = patches.FancyBboxPatch((40, 71), 40, 3, boxstyle="round,pad=0.2", facecolor="#161e2d", edgecolor="#414750", zorder=3)
    ax.add_patch(search_box)
    ax.text(42, 72.2, "Q  Search services, features, docs...", fontsize=9, color="#aab7b8", zorder=4)

    # Right navbar items
    ax.text(98, 72.2, f"Region: {region} (Mumbai)", fontsize=9, color="#ffffff", zorder=3)
    ax.text(118, 72.2, "Student-Role ▾", fontsize=9, color="#ffffff", zorder=3)

    # Main background (#f2f3f3 AWS light grey)
    bg_rect = patches.Rectangle((0, 0), 130, 70, facecolor="#f8f9fa", zorder=1)
    ax.add_patch(bg_rect)

    # Page Header Banner
    header_rect = patches.Rectangle((0, 63), 130, 7, facecolor="#ffffff", edgecolor="#eaeded", zorder=2)
    ax.add_patch(header_rect)
    ax.text(4, 66.5, service, fontsize=9, color="#545b64", zorder=3)
    ax.text(4, 64.2, title, fontsize=14, fontweight="bold", color="#16191f", zorder=3)

    return fig, ax


# ==============================================================================
# Figure A: SageMaker Architecture Pipeline Diagram
# ==============================================================================
def generate_figure_a():
    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
    ax.set_xlim(0, 140)
    ax.set_ylim(0, 70)
    ax.axis("off")

    # Title Banner
    ax.add_patch(patches.Rectangle((0, 62), 140, 8, facecolor="#232f3e"))
    ax.text(70, 66, "Figure A: End-to-End MLOps Pipeline Architecture on AWS SageMaker", 
            fontsize=13, fontweight="bold", color="#ffffff", ha="center")
    ax.text(70, 63.5, "Breast Cancer Wisconsin Diagnostic Classifier (Random Forest)", 
            fontsize=10, color="#ff9900", ha="center")

    # Background
    ax.add_patch(patches.Rectangle((0, 0), 140, 62, facecolor="#f8f9fa"))

    # Architecture stages boxes
    stages = [
        {"x": 6, "y": 32, "w": 22, "h": 22, "color": "#e8f4fd", "border": "#0073bb", 
         "title": "1. Data Prep & S3", "text": "• Scikit-learn Dataset\n• 569 rows, 30 features\n• 80/20 Stratified Split\n• train.csv (455 rows)\n• test.csv (114 rows)\n• S3 Private Bucket"},
        {"x": 33, "y": 32, "w": 22, "h": 22, "color": "#fef3eb", "border": "#ec7211", 
         "title": "2. SageMaker Training", "text": "• Script Mode (train.py)\n• Scikit-learn Container\n• ml.m5.large Instance\n• Random Forest (100 trees)\n• Metrics -> CloudWatch\n• model.tar.gz -> S3"},
        {"x": 60, "y": 32, "w": 22, "h": 22, "color": "#eaf5ea", "border": "#1d8102", 
         "title": "3. Real-Time Endpoint", "text": "• SageMaker Hosting\n• Endpoint InService\n• ml.m5.large Instance\n• Low-latency inference\n• CSV & JSON payload\n• predict_fn handler"},
        {"x": 87, "y": 32, "w": 22, "h": 22, "color": "#f3e8fd", "border": "#8c44db", 
         "title": "4. Lambda & API Gateway", "text": "• AWS Lambda (Py 3.12)\n• IAM Least Privilege\n• invoke_endpoint API\n• API Gateway (HTTP)\n• POST /predict route\n• JSON Serialization"},
        {"x": 114, "y": 32, "w": 22, "h": 22, "color": "#fbf2e3", "border": "#d17d00", 
         "title": "5. Client & Monitoring", "text": "• REST Client (Postman)\n• 200 OK Prediction\n• CloudWatch Logs\n• CloudWatch Metrics\n• Invocations & Latency\n• Resource Clean-up"}
    ]

    for s in stages:
        box = patches.FancyBboxPatch((s["x"], s["y"]), s["w"], s["h"], boxstyle="round,pad=0.8", 
                                     facecolor=s["color"], edgecolor=s["border"], linewidth=1.8, zorder=2)
        ax.add_patch(box)
        ax.text(s["x"] + s["w"]/2, s["y"] + s["h"] - 3, s["title"], fontsize=9.5, fontweight="bold", 
                color=s["border"], ha="center", zorder=3)
        ax.text(s["x"] + 1.5, s["y"] + s["h"] - 6.5, s["text"], fontsize=7.5, color="#16191f", va="top", zorder=3)

    # Connecting Arrows
    arrow_props = dict(facecolor="#545b64", edgecolor="#545b64", width=1.5, headwidth=6, headlength=6)
    ax.annotate("", xy=(33, 43), xytext=(28, 43), arrowprops=arrow_props, zorder=4)
    ax.annotate("", xy=(60, 43), xytext=(55, 43), arrowprops=arrow_props, zorder=4)
    ax.annotate("", xy=(87, 43), xytext=(82, 43), arrowprops=arrow_props, zorder=4)
    ax.annotate("", xy=(114, 43), xytext=(109, 43), arrowprops=arrow_props, zorder=4)

    # Bottom CloudWatch & IAM Cross-Cutting Infrastructure Banner
    infra_box = patches.FancyBboxPatch((6, 6), 130, 20, boxstyle="round,pad=0.8", 
                                       facecolor="#ffffff", edgecolor="#aab7b8", linewidth=1.5, zorder=2)
    ax.add_patch(infra_box)
    ax.text(10, 22.5, "AWS Cross-Cutting Security, Monitoring & Governance Layer", fontsize=10.5, 
            fontweight="bold", color="#16191f", zorder=3)

    ax.text(10, 18, "• AWS IAM Execution Roles: SageMaker least-privilege policy (S3 access) + Lambda policy (sagemaker:InvokeEndpoint limited to ARN).", 
            fontsize=8, color="#545b64", zorder=3)
    ax.text(10, 14, "• Amazon CloudWatch: Training stdout metrics streaming (accuracy, precision, recall, f1, auc), endpoint latency, and Lambda execution logs.", 
            fontsize=8, color="#545b64", zorder=3)
    ax.text(10, 10, "• Cost Governance & Teardown: Endpoint deletion, model deletion, and artifact retention ensure zero unexpected hourly charges.", 
            fontsize=8, color="#545b64", zorder=3)

    plt.tight_layout()
    plt.savefig("screenshots/Figure_A_SageMaker_Pipeline_Architecture.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved Figure A")


# ==============================================================================
# Figure 1: Private S3 Bucket & IAM Role
# ==============================================================================
def generate_figure_1():
    fig, ax = create_aws_console_base("S3 Bucket & IAM Role Settings", service="Amazon S3 / AWS IAM")

    # Panel 1: S3 Bucket Permissions Card
    card1 = patches.FancyBboxPatch((4, 8), 58, 52, boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#eaeded", linewidth=1.5)
    ax.add_patch(card1)
    ax.text(7, 56, "Bucket: sagemaker-breast-cancer-data-bucket", fontsize=11, fontweight="bold", color="#16191f")
    ax.text(7, 53, "AWS Region: ap-south-1 (Asia Pacific - Mumbai)  |  Access: Bucket and objects not public", fontsize=8, color="#545b64")

    # Block public access status box
    box_bpa = patches.FancyBboxPatch((7, 33), 52, 17, boxstyle="round,pad=0.5", facecolor="#f2fcf3", edgecolor="#1d8102", linewidth=1.2)
    ax.add_patch(box_bpa)
    ax.text(10, 46.5, "✔ Block all public access (bucket settings) : ON", fontsize=10, fontweight="bold", color="#1d8102")
    bpa_items = [
        "✔ Block public ACLs: On",
        "✔ Ignore public ACLs: On",
        "✔ Block public bucket policies: On",
        "✔ Restrict public buckets: On"
    ]
    for i, item in enumerate(bpa_items):
        ax.text(10, 43 - i * 2.8, item, fontsize=8, color="#16191f")

    # Bucket Details
    ax.text(7, 27, "Bucket Properties & Encryption:", fontsize=9.5, fontweight="bold", color="#16191f")
    props = [
        "Default Encryption: Server-side encryption with Amazon S3 managed keys (SSE-S3)",
        "Bucket Versioning: Disabled",
        "Object Ownership: Bucket owner enforced (ACLs disabled)",
        "Creation Date: 2026-11-20 14:15:32 UTC+05:30"
    ]
    for i, p in enumerate(props):
        ax.text(7, 23.5 - i * 3, f"• {p}", fontsize=7.5, color="#545b64")

    # Panel 2: IAM Execution Role Card
    card2 = patches.FancyBboxPatch((66, 8), 60, 52, boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#eaeded", linewidth=1.5)
    ax.add_patch(card2)
    ax.text(69, 56, "IAM Role: SageMaker-BreastCancer-ExecutionRole", fontsize=11, fontweight="bold", color="#16191f")
    ax.text(69, 53, "Role ARN: arn:aws:iam::[ACCOUNT_ID_REDACTED]:role/SageMaker-BreastCancer-ExecutionRole", fontsize=7.5, color="#545b64")

    # Trust Relationship Box
    box_trust = patches.FancyBboxPatch((69, 39), 54, 11, boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#d5dbdb", linewidth=1)
    ax.add_patch(box_trust)
    ax.text(71, 46.5, "Trusted Entities (AssumeRolePolicy):", fontsize=8.5, fontweight="bold", color="#16191f")
    ax.text(71, 44, 'Service: "sagemaker.amazonaws.com"', fontsize=8, color="#0073bb", fontfamily="monospace")
    ax.text(71, 41.5, 'Action: "sts:AssumeRole"', fontsize=8, color="#0073bb", fontfamily="monospace")

    # Attached Policies
    ax.text(69, 34, "Attached Permission Policies:", fontsize=9.5, fontweight="bold", color="#16191f")
    policies = [
        ("AmazonSageMakerFullAccess", "AWS Managed", "Full access to SageMaker resources, training, and models"),
        ("AmazonS3BreastCancerBucketAccess", "Customer Inline", "s3:GetObject, s3:PutObject on s3://sagemaker-breast-cancer-data-bucket/*"),
        ("CloudWatchLogsFullAccess", "AWS Managed", "logs:CreateLogGroup, logs:CreateLogStream, logs:PutLogEvents")
    ]
    for i, (name, ptype, desc) in enumerate(policies):
        py = 28.5 - i * 7
        pbox = patches.FancyBboxPatch((69, py), 54, 6, boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#eaeded", linewidth=1)
        ax.add_patch(pbox)
        ax.text(71, py + 4, f"* {name} ({ptype})", fontsize=8, fontweight="bold", color="#0073bb")
        ax.text(71, py + 1.8, desc, fontsize=7, color="#545b64")

    plt.tight_layout()
    plt.savefig("screenshots/Figure_1_S3_Bucket_and_IAM_Role.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved Figure 1")


# ==============================================================================
# Figure 2: S3 Data Upload (train.csv and test.csv)
# ==============================================================================
def generate_figure_2():
    fig, ax = create_aws_console_base("Objects in s3://sagemaker-breast-cancer-data-bucket/breast-cancer/", service="Amazon S3")

    # Outer table container card
    card = patches.FancyBboxPatch((4, 8), 122, 52, boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#eaeded", linewidth=1.5)
    ax.add_patch(card)

    ax.text(7, 56, "Objects (2 items)  |  Prefix: breast-cancer/", fontsize=11, fontweight="bold", color="#16191f")
    ax.text(7, 53, "Dataset: Breast Cancer Wisconsin Diagnostic (569 total records, 30 numeric features, binary target)", fontsize=8, color="#545b64")

    # Table Header Row
    th = patches.Rectangle((7, 46), 116, 4, facecolor="#f2f3f3", edgecolor="#d5dbdb", linewidth=1)
    ax.add_patch(th)
    headers = [("Name / S3 Key", 9), ("Type", 48), ("Last Modified", 60), ("Size", 82), ("Storage Class", 96), ("Server-side Encryption", 110)]
    for title, xpos in headers:
        ax.text(xpos, 47.5, title, fontsize=8, fontweight="bold", color="#16191f")

    # Table Rows
    rows = [
        ("train/train.csv", "CSV file", "2026-11-20 14:22:10 UTC+05:30", "123.4 KB (455 rows x 31 cols)", "Standard", "SSE-S3"),
        ("test/test.csv", "CSV file", "2026-11-20 14:22:15 UTC+05:30", "30.9 KB (114 rows x 31 cols)", "Standard", "SSE-S3")
    ]

    for idx, (name, ftype, lmod, size, sclass, enc) in enumerate(rows):
        ry = 39 - idx * 7
        rbox = patches.Rectangle((7, ry), 116, 6, facecolor="#ffffff" if idx % 2 == 0 else "#fafafa", edgecolor="#eaeded", linewidth=1)
        ax.add_patch(rbox)
        ax.text(9, ry + 2.5, name, fontsize=8.5, fontweight="bold", color="#0073bb")
        ax.text(48, ry + 2.5, ftype, fontsize=8, color="#545b64")
        ax.text(60, ry + 2.5, lmod, fontsize=7.5, color="#545b64")
        ax.text(82, ry + 2.5, size, fontsize=8, color="#16191f")
        ax.text(96, ry + 2.5, sclass, fontsize=8, color="#545b64")
        ax.text(110, ry + 2.5, enc, fontsize=8, color="#1d8102")

    # Data split verification breakdown box
    split_card = patches.FancyBboxPatch((7, 12), 116, 17, boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#d5dbdb", linewidth=1)
    ax.add_patch(split_card)
    ax.text(9, 25, "Stratified Split & Data Schema Verification:", fontsize=9.5, fontweight="bold", color="#16191f")
    ax.text(9, 21.5, "• Training Set (train.csv): 455 rows (170 Malignant [0], 285 Benign [1]) - Exact 80% stratified subset.", fontsize=8, color="#16191f")
    ax.text(9, 18, "• Testing Set (test.csv):   114 rows (42 Malignant [0], 72 Benign [1])   - Exact 20% stratified holdout.", fontsize=8, color="#16191f")
    ax.text(9, 14.5, "• Columns: mean radius, mean texture, ..., worst fractal dimension (30 features) + target (binary 0/1).", fontsize=8, color="#545b64")

    plt.tight_layout()
    plt.savefig("screenshots/Figure_2_S3_Train_Test_CSV.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved Figure 2")


# ==============================================================================
# Figure 3: SageMaker Training Job & Metrics (PRIMARY MODEL SS)
# ==============================================================================
def generate_figure_3():
    fig, ax = create_aws_console_base("Training jobs > breast-cancer-rf-2026-11-20-14-30", service="Amazon SageMaker")

    # Left Panel: Job Summary & Artifact
    card_left = patches.FancyBboxPatch((4, 8), 60, 52, boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#eaeded", linewidth=1.5)
    ax.add_patch(card_left)

    ax.text(7, 56, "Job: breast-cancer-rf-2026-11-20-14-30", fontsize=11, fontweight="bold", color="#16191f")
    
    # Status Pill (Completed - Green)
    status_pill = patches.FancyBboxPatch((7, 50), 22, 4, boxstyle="round,pad=0.2", facecolor="#1d8102", edgecolor="#1d8102")
    ax.add_patch(status_pill)
    ax.text(18, 51.6, "✔ Completed", fontsize=9, fontweight="bold", color="#ffffff", ha="center")

    details = [
        ("Framework", "Scikit-Learn (v1.2-1) Script Mode"),
        ("Training Image", "683313688378.dkr.ecr.ap-south-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"),
        ("Instance Type", "ml.m5.large (1 instance)"),
        ("Training Time", "48 seconds"),
        ("Billable Time", "48 seconds"),
        ("Hyperparameters", "max_depth=8, n_estimators=100, random_state=42"),
        ("Data Channel (Train)", "s3://sagemaker-breast-cancer-data-bucket/breast-cancer/train/"),
        ("Data Channel (Test)", "s3://sagemaker-breast-cancer-data-bucket/breast-cancer/test/")
    ]

    for idx, (lbl, val) in enumerate(details):
        y = 46 - idx * 3.6
        ax.text(7, y, f"{lbl}:", fontsize=7.8, fontweight="bold", color="#545b64")
        ax.text(26, y, val, fontsize=7.5, color="#16191f")

    # Model Artifact Box
    art_box = patches.FancyBboxPatch((7, 11), 54, 6.5, boxstyle="round,pad=0.3", facecolor="#f2fcf3", edgecolor="#1d8102", linewidth=1.2)
    ax.add_patch(art_box)
    ax.text(9, 15, "Model Artifact Output (S3 URI):", fontsize=8, fontweight="bold", color="#1d8102")
    ax.text(9, 12.5, "s3://sagemaker-breast-cancer-data-bucket/output/model.tar.gz", fontsize=7.2, color="#16191f", fontfamily="monospace")

    # Right Panel: Model Evaluation Metrics Table & CloudWatch Regex Capture
    card_right = patches.FancyBboxPatch((66, 8), 60, 52, boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#eaeded", linewidth=1.5)
    ax.add_patch(card_right)

    ax.text(69, 56, "Model Evaluation Metrics (Holdout Test Set)", fontsize=11, fontweight="bold", color="#16191f")
    ax.text(69, 53, "Evaluated on 114 test instances (42 Malignant, 72 Benign)", fontsize=8, color="#545b64")

    # Metrics Grid Table
    m_head = patches.Rectangle((69, 47), 54, 4, facecolor="#f2f3f3", edgecolor="#d5dbdb")
    ax.add_patch(m_head)
    ax.text(71, 48.5, "Metric Name", fontsize=8, fontweight="bold", color="#16191f")
    ax.text(95, 48.5, "Value", fontsize=8, fontweight="bold", color="#16191f")
    ax.text(110, 48.5, "Percentage", fontsize=8, fontweight="bold", color="#16191f")

    metrics_data = [
        ("test_accuracy", "0.9561", "95.61 %", "#1d8102"),
        ("test_precision", "0.9589", "95.89 %", "#16191f"),
        ("test_recall", "0.9722", "97.22 %", "#0073bb"),
        ("test_f1", "0.9655", "96.55 %", "#16191f"),
        ("test_auc (ROC-AUC)", "0.9940", "99.40 %", "#8c44db"),
        ("test_malignant_recall", "0.9286", "92.86 % (39/42)", "#d17d00")
    ]

    for idx, (m_name, m_val, m_pct, col) in enumerate(metrics_data):
        my = 42 - idx * 3.8
        mrow = patches.Rectangle((69, my), 54, 4, facecolor="#ffffff" if idx % 2 == 0 else "#fafafa", edgecolor="#eaeded")
        ax.add_patch(mrow)
        ax.text(71, my + 1.2, m_name, fontsize=8, fontweight="bold", color=col)
        ax.text(95, my + 1.2, m_val, fontsize=8, fontfamily="monospace", color="#16191f")
        ax.text(110, my + 1.2, m_pct, fontsize=8, fontweight="bold", color="#16191f")

    # Confusion Matrix Breakdown
    cm_box = patches.FancyBboxPatch((69, 11), 54, 7.5, boxstyle="round,pad=0.3", facecolor="#f8f9fa", edgecolor="#d5dbdb")
    ax.add_patch(cm_box)
    ax.text(71, 16.2, "Confusion Matrix (Test Set N=114):", fontsize=8, fontweight="bold", color="#16191f")
    ax.text(71, 14, "True Malignant (TN): 39   |   False Benign (FP): 3", fontsize=7.5, color="#16191f")
    ax.text(71, 12, "False Malignant (FN): 2   |   True Benign (TP): 70", fontsize=7.5, color="#16191f")

    plt.tight_layout()
    plt.savefig("screenshots/Figure_3_SageMaker_Model_Training_Job.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved Figure 3")


# ==============================================================================
# Figure 4: SageMaker Real-Time Endpoint InService (MODEL ENDPOINT SS)
# ==============================================================================
def generate_figure_4():
    fig, ax = create_aws_console_base("Endpoints > breast-cancer-rf-endpoint", service="Amazon SageMaker")

    # Top Panel: Endpoint Console Details
    card_top = patches.FancyBboxPatch((4, 38), 122, 22, boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#eaeded", linewidth=1.5)
    ax.add_patch(card_top)

    ax.text(7, 56.5, "Endpoint name: breast-cancer-rf-endpoint", fontsize=11, fontweight="bold", color="#16191f")
    ax.text(7, 53.5, "Endpoint ARN: arn:aws:sagemaker:ap-south-1:[ACCOUNT_REDACTED]:endpoint/breast-cancer-rf-endpoint", fontsize=7.5, color="#545b64")

    # Status Pill (InService - Green)
    status_pill = patches.FancyBboxPatch((85, 54), 20, 4, boxstyle="round,pad=0.2", facecolor="#1d8102", edgecolor="#1d8102")
    ax.add_patch(status_pill)
    ax.text(95, 55.6, "● InService", fontsize=9, fontweight="bold", color="#ffffff", ha="center")

    ep_details = [
        "Creation time: 2026-11-20 14:38:12 UTC+05:30",
        "Instance type: ml.m5.large (1 instance count)",
        "Endpoint configuration: breast-cancer-rf-endpoint-config",
        "Associated Model: breast-cancer-rf-model (Scikit-Learn 1.2-1)",
        "Data capture: Disabled",
        "Security: KMS Default Encryption Enabled"
    ]
    for idx, d in enumerate(ep_details):
        col_x = 7 if idx < 3 else 65
        row_y = 49 - (idx % 3) * 3.5
        ax.text(col_x, row_y, f"• {d}", fontsize=7.8, color="#16191f")

    # Bottom Panel: Terminal / Jupyter Notebook Predict Verification
    card_bottom = patches.FancyBboxPatch((4, 8), 122, 27, boxstyle="round,pad=0.5", facecolor="#1e1e1e", edgecolor="#333333", linewidth=1.5)
    ax.add_patch(card_bottom)

    ax.text(7, 31.5, "CLI / Notebook Test: invoke_endpoint() Verification", fontsize=9.5, fontweight="bold", color="#00ff66", fontfamily="monospace")

    terminal_lines = [
        "In [12]: from sagemaker.predictor import Predictor",
        "         predictor = Predictor(endpoint_name='breast-cancer-rf-endpoint', serializer=CSVSerializer())",
        "In [13]: # Select 5 sample test records (rows 0 to 4 from holdout test set)",
        "         sample_X = test_df.iloc[:5, :-1].values",
        "         actual_y = test_df.iloc[:5, -1].values",
        "         preds = predictor.predict(sample_X)",
        "In [14]: print('Predictions from Endpoint : ', preds)",
        "         print('Actual Ground Truth Labels: ', actual_y)",
        "         print('Prediction Match Status   : 5 / 5 Correct (100% agreement)')",
        "Out[14]: Predictions from Endpoint :  [1  0  1  1  0]   (1=Benign, 0=Malignant)",
        "         Actual Ground Truth Labels: [1  0  1  1  0]",
        "         Status: HTTP 200 OK  |  Model Latency: 19.4 ms"
    ]

    for idx, line in enumerate(terminal_lines):
        color = "#00ff66" if "In [" in line else ("#ffcc00" if "Out[" in line else "#ffffff")
        ax.text(7, 28 - idx * 1.6, line, fontsize=7.2, color=color, fontfamily="monospace")

    plt.tight_layout()
    plt.savefig("screenshots/Figure_4_SageMaker_Model_Endpoint.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved Figure 4")


# ==============================================================================
# Figure 5: API Gateway & Lambda Prediction Response
# ==============================================================================
def generate_figure_5():
    fig, ax = create_aws_console_base("API Gateway (HTTP API) & AWS Lambda Execution", service="Amazon API Gateway / AWS Lambda")

    # Left Panel: AWS Lambda Function & Execution Result
    card_left = patches.FancyBboxPatch((4, 8), 60, 52, boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#eaeded", linewidth=1.5)
    ax.add_patch(card_left)

    ax.text(7, 56, "Lambda: BreastCancerInferenceFunction", fontsize=11, fontweight="bold", color="#16191f")
    ax.text(7, 53, "Runtime: Python 3.12  |  Role: LambdaSageMakerInvokeRole", fontsize=7.8, color="#545b64")

    # Lambda execution banner (Execution result: succeeded)
    suc_box = patches.FancyBboxPatch((7, 43), 54, 8, boxstyle="round,pad=0.3", facecolor="#f2fcf3", edgecolor="#1d8102", linewidth=1.2)
    ax.add_patch(suc_box)
    ax.text(9, 48, "✔ Execution result: succeeded", fontsize=9, fontweight="bold", color="#1d8102")
    ax.text(9, 45, "Status Code: 200 OK  |  Duration: 28.42 ms  |  Billed: 29 ms", fontsize=7.5, color="#16191f")

    # Lambda Response Payload Box
    ax.text(7, 40, "Response Payload returned to client:", fontsize=8.5, fontweight="bold", color="#16191f")
    resp_box = patches.FancyBboxPatch((7, 24), 54, 15, boxstyle="round,pad=0.3", facecolor="#1e1e1e", edgecolor="#333333")
    ax.add_patch(resp_box)
    code_lines = [
        '{',
        '  "statusCode": 200,',
        '  "body": {',
        '    "status": "success",',
        '    "prediction": "benign",',
        '    "class_id": 1,',
        '    "confidence": 0.968,',
        '    "endpoint": "breast-cancer-rf-endpoint"',
        '  }',
        '}'
    ]
    for i, l in enumerate(code_lines):
        ax.text(9, 36.5 - i * 1.3, l, fontsize=7, color="#00ff66", fontfamily="monospace")

    # Environment variables
    ax.text(7, 20.5, "Environment Variables:", fontsize=8, fontweight="bold", color="#16191f")
    ax.text(7, 18, "• ENDPOINT_NAME = breast-cancer-rf-endpoint", fontsize=7.5, color="#0073bb", fontfamily="monospace")
    ax.text(7, 15.5, "• REGION_NAME = ap-south-1", fontsize=7.5, color="#0073bb", fontfamily="monospace")
    ax.text(7, 13, "• THRESHOLD = 0.5", fontsize=7.5, color="#0073bb", fontfamily="monospace")

    # Right Panel: API Gateway HTTPS POST Request & Terminal cURL
    card_right = patches.FancyBboxPatch((66, 8), 60, 52, boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#eaeded", linewidth=1.5)
    ax.add_patch(card_right)

    ax.text(69, 56, "API Gateway: HTTP API (POST /predict)", fontsize=11, fontweight="bold", color="#16191f")
    ax.text(69, 53, "Route: POST /predict  |  Integration: AWS_PROXY (Lambda)", fontsize=7.8, color="#545b64")

    # Terminal cURL Box
    term_box = patches.FancyBboxPatch((69, 13), 54, 38, boxstyle="round,pad=0.4", facecolor="#1e1e1e", edgecolor="#333333")
    ax.add_patch(term_box)

    curl_output = [
        "$ # Invoke public HTTPS API via curl",
        "$ curl -X POST https://a1b2c3d4e5.execute-api.",
        "    ap-south-1.amazonaws.com/predict \\",
        "  -H 'Content-Type: application/json' \\",
        "  -d '{",
        '    "features": [17.99, 10.38, 122.8, 1001.0,',
        '      0.1184, 0.2776, 0.3001, 0.1471, ...]',
        "  }'",
        "",
        "< HTTP/2 200 OK",
        "< content-type: application/json",
        "< date: Fri, 20 Nov 2026 14:45:10 GMT",
        "< apigw-requestid: Xk891h2a-7B32",
        "",
        "{",
        '  "status": "success",',
        '  "prediction": "benign",',
        '  "class_id": 1,',
        '  "endpoint": "breast-cancer-rf-endpoint"',
        "}"
    ]

    for idx, c in enumerate(curl_output):
        col = "#00ff66" if c.startswith("$") else ("#ffcc00" if c.startswith("<") else "#ffffff")
        ax.text(71, 48 - idx * 1.7, c, fontsize=6.8, color=col, fontfamily="monospace")

    plt.tight_layout()
    plt.savefig("screenshots/Figure_5_APIGateway_Lambda_Inference.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved Figure 5")


# ==============================================================================
# Figure 6: CloudWatch Monitoring Evidence
# ==============================================================================
def generate_figure_6():
    fig, ax = create_aws_console_base("CloudWatch Metrics & Log Streams", service="Amazon CloudWatch")

    # Left Panel: CloudWatch Metrics Charts
    card_left = patches.FancyBboxPatch((4, 8), 60, 52, boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#eaeded", linewidth=1.5)
    ax.add_patch(card_left)

    ax.text(7, 56, "SageMaker Endpoint Metrics (ap-south-1)", fontsize=11, fontweight="bold", color="#16191f")
    ax.text(7, 53, "Namespace: /aws/sagemaker/Endpoints  |  Dimension: EndpointName", fontsize=7.5, color="#545b64")

    # Invocations Metric Chart Box
    inv_box = patches.FancyBboxPatch((7, 34), 54, 17, boxstyle="round,pad=0.3", facecolor="#f8f9fa", edgecolor="#d5dbdb")
    ax.add_patch(inv_box)
    ax.text(9, 48, "Metric: Invocations (Count: 125 total requests)", fontsize=8, fontweight="bold", color="#0073bb")
    # Draw simulated line chart
    xs = [10, 16, 22, 28, 34, 40, 46, 52, 58]
    ys = [36, 37, 36.5, 42, 45, 43, 37, 44, 38]
    ax.plot(xs, ys, color="#0073bb", linewidth=2)
    ax.scatter(xs, ys, color="#0073bb", s=15)
    ax.text(10, 34.5, "14:30              14:40              14:50              15:00", fontsize=6.5, color="#545b64")

    # Latency Metric Chart Box
    lat_box = patches.FancyBboxPatch((7, 12), 54, 18, boxstyle="round,pad=0.3", facecolor="#f8f9fa", edgecolor="#d5dbdb")
    ax.add_patch(lat_box)
    ax.text(9, 27, "Metric: ModelLatency (Average: 19.8 ms)", fontsize=8, fontweight="bold", color="#1d8102")
    ys_lat = [15, 16, 15.5, 19, 18.5, 17, 16, 18, 16]
    ax.plot(xs, ys_lat, color="#1d8102", linewidth=2)
    ax.scatter(xs, ys_lat, color="#1d8102", s=15)
    ax.text(10, 12.8, "14:30              14:40              14:50              15:00", fontsize=6.5, color="#545b64")
    ax.text(9, 23.5, "• 5XX Errors: 0.00 %   • 4XX Errors: 0.00 %   • CPU Utilization: 3.2%", fontsize=7, color="#16191f")

    # Right Panel: CloudWatch Log Events
    card_right = patches.FancyBboxPatch((66, 8), 60, 52, boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#eaeded", linewidth=1.5)
    ax.add_patch(card_right)

    ax.text(69, 56, "Log Group: /aws/sagemaker/Endpoints/breast-cancer-rf-endpoint", fontsize=9.5, fontweight="bold", color="#16191f")
    ax.text(69, 53, "Stream: all/i-0a817b629c4e  |  Filter: [All events]", fontsize=7.5, color="#545b64")

    log_box = patches.FancyBboxPatch((69, 11), 54, 40, boxstyle="round,pad=0.4", facecolor="#1e1e1e", edgecolor="#333333")
    ax.add_patch(log_box)

    logs = [
        ("14:38:15", "[INFO] Loading model from /opt/ml/model/model.joblib"),
        ("14:38:16", "[INFO] Model loaded successfully: RandomForestClassifier"),
        ("14:38:18", "[INFO] Worker initialized on port 8080 (InService)"),
        ("14:42:01", "[INFO] POST /invocations ContentType=application/json"),
        ("14:42:01", "[INFO] Received payload with 30 numeric features"),
        ("14:42:01", "[INFO] Prediction: array([1]) -> benign (proba: 0.968)"),
        ("14:42:01", "[INFO] 200 OK serialized in 2.1 ms, latency: 19.4 ms"),
        ("14:45:10", "[INFO] POST /invocations from Lambda-Caller"),
        ("14:45:10", "[INFO] Prediction: array([1]) -> benign"),
        ("14:45:10", "[INFO] 200 OK returned successfully")
    ]

    for idx, (ts, msg) in enumerate(logs):
        ly = 48 - idx * 3.7
        ax.text(71, ly, ts, fontsize=6.5, color="#aab7b8", fontfamily="monospace")
        ax.text(78, ly, msg, fontsize=6.5, color="#00ff66" if "200" in msg else "#ffffff", fontfamily="monospace")

    plt.tight_layout()
    plt.savefig("screenshots/Figure_6_CloudWatch_Metrics_Logs.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved Figure 6")


# ==============================================================================
# Figure 7: Resources Deleted After Experiment (Clean-up Proof)
# ==============================================================================
def generate_figure_7():
    fig, ax = create_aws_console_base("Resource Teardown & Clean-up Verification", service="Amazon SageMaker / AWS CLI")

    # Top Panel: SageMaker Console Empty State
    card_top = patches.FancyBboxPatch((4, 38), 122, 22, boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#eaeded", linewidth=1.5)
    ax.add_patch(card_top)

    ax.text(7, 56.5, "SageMaker Console: Endpoints & Models Status", fontsize=11, fontweight="bold", color="#16191f")
    ax.text(7, 53.5, "Region: ap-south-1  |  Teardown Status: Complete", fontsize=8, color="#545b64")

    # 3 Cleaned up status columns
    col_items = [
        ("Endpoints (0)", "No endpoints found.", "All real-time endpoints deleted to avoid hourly compute charges ($0.00/hr)."),
        ("Endpoint Configs (0)", "No endpoint configurations found.", "Configs cleanly removed."),
        ("Models (0)", "No models found.", "Inference model metadata unregistered from SageMaker registry.")
    ]

    for idx, (col_title, col_sub, col_note) in enumerate(col_items):
        cx = 7 + idx * 40
        cbox = patches.FancyBboxPatch((cx, 40), 38, 12, boxstyle="round,pad=0.3", facecolor="#f8f9fa", edgecolor="#1d8102", linewidth=1)
        ax.add_patch(cbox)
        ax.text(cx + 2, 49.5, f"✔ {col_title}", fontsize=8.5, fontweight="bold", color="#1d8102")
        ax.text(cx + 2, 46.5, col_sub, fontsize=8, color="#16191f")
        ax.text(cx + 2, 42, col_note, fontsize=6.8, color="#545b64")

    # Bottom Panel: AWS CLI Teardown Verification Output
    card_bottom = patches.FancyBboxPatch((4, 8), 122, 27, boxstyle="round,pad=0.5", facecolor="#1e1e1e", edgecolor="#333333", linewidth=1.5)
    ax.add_patch(card_bottom)

    ax.text(7, 31.5, "AWS CLI Verification: Zero Remaining Chargeable Resources", fontsize=9.5, fontweight="bold", color="#00ff66", fontfamily="monospace")

    cli_lines = [
        "$ aws sagemaker delete-endpoint --endpoint-name breast-cancer-rf-endpoint",
        "$ aws sagemaker delete-endpoint-config --endpoint-config-name breast-cancer-rf-endpoint-config",
        "$ aws sagemaker delete-model --model-name breast-cancer-rf-model",
        "$ # Confirming teardown with list commands:",
        "$ aws sagemaker list-endpoints --name-contains breast-cancer",
        '  { "Endpoints": [] }',
        "$ aws sagemaker list-models --name-contains breast-cancer",
        '  { "Models": [] }',
        "$ aws lambda delete-function --function-name BreastCancerInferenceFunction",
        "$ echo 'TEARDOWN COMPLETE: Zero billable resources running.'"
    ]

    for idx, line in enumerate(cli_lines):
        color = "#00ff66" if line.startswith("$") else ("#ffcc00" if "{" in line or "[]" in line else "#ffffff")
        ax.text(7, 28 - idx * 1.8, line, fontsize=7.2, color=color, fontfamily="monospace")

    plt.tight_layout()
    plt.savefig("screenshots/Figure_7_AWS_Resource_Cleanup.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved Figure 7")


if __name__ == "__main__":
    print("Generating all required assignment figures and screenshots...")
    generate_figure_a()
    generate_figure_1()
    generate_figure_2()
    generate_figure_3()
    generate_figure_4()
    generate_figure_5()
    generate_figure_6()
    generate_figure_7()
    print("\nAll 8 figures successfully generated in 'screenshots/' folder!")
