import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_architecture_diagram(output_path="report/architecture_diagram.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Background canvas
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")

    # Helper function for styled boxes
    def draw_box(x, y, w, h, title, subtitle, bg_color, border_color):
        box = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.15",
            facecolor=bg_color, edgecolor=border_color, linewidth=2, zorder=2
        )
        ax.add_patch(box)
        ax.text(x + w/2, y + h*0.62, title, color="white", fontsize=11, fontweight="bold",
                ha="center", va="center", zorder=3)
        ax.text(x + w/2, y + h*0.32, subtitle, color="#94A3B8", fontsize=8.5,
                ha="center", va="center", zorder=3)

    # Helper for styled arrows
    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.45",
                            color="#38BDF8", lw=2, shrinkA=4, shrinkB=4),
            zorder=4
        )
        if label:
            mx, my = (x1 + x2)/2, (y1 + y2)/2 + 0.18
            ax.text(mx, my, label, color="#F8FAFC", fontsize=8, fontweight="semibold",
                    ha="center", va="center", bbox=dict(boxstyle="round,pad=0.2", facecolor="#1E293B", edgecolor="#475569", lw=0.8), zorder=5)

    # 1. Traffic Generator / Client
    draw_box(0.5, 4.2, 2.2, 1.4, "Traffic Simulator\n(Client)", "simulate.py\nNormal / Drift / Errors", "#1E293B", "#38BDF8")

    # 2. FastAPI Model Service
    draw_box(3.6, 3.8, 2.7, 2.2, "FastAPI ML Service", "model-service:8000\n/predict, /feedback\n/health, /metrics\nIn-memory Drift Tracker", "#1E293B", "#10B981")

    # 2b. Model Artifacts
    draw_box(3.8, 1.2, 2.3, 1.2, "Model Storage", "model.joblib\ntraining_stats.json", "#1E293B", "#64748B")

    # 3. Node Exporter
    draw_box(3.6, 0.1, 2.7, 0.9, "Node Exporter", "node-exporter:9100\nHost CPU & RAM metrics", "#1E293B", "#F59E0B")

    # 4. Prometheus Server
    draw_box(7.2, 3.2, 2.5, 2.6, "Prometheus Server", "ml-prometheus:9090\nScrape Engine (5s)\nTSDB Storage\n5 Alert Evaluation Rules", "#1E293B", "#F97316")

    # 5. Alertmanager
    draw_box(7.2, 0.8, 2.5, 1.3, "Alertmanager", "ml-alertmanager:9093\nAlert Deduplication\nRouting & Notifications", "#1E293B", "#EF4444")

    # 6. Grafana Visualization
    draw_box(10.2, 3.5, 1.6, 2.0, "Grafana", "ml-grafana:3001\nService Row\nModel Row\nInfra Row", "#1E293B", "#A855F7")

    # Connectors
    draw_arrow(2.7, 4.9, 3.6, 4.9, "HTTP POST\nInference & Feedback")
    draw_arrow(4.9, 2.4, 4.9, 3.8, "Load on Startup")
    draw_arrow(6.3, 5.0, 7.2, 5.0, "Pull Scrape /metrics (5s)")
    draw_arrow(6.3, 0.55, 7.2, 3.6, "Scrape Host (5s)")
    draw_arrow(8.45, 3.2, 8.45, 2.1, "Push Triggered Alerts")
    draw_arrow(9.7, 4.5, 10.2, 4.5, "PromQL Queries\n(Datasource proxy)")

    # Title header
    plt.title("MLOps Telemetry & Observability Architecture (Assignment 8)",
              color="white", fontsize=14, fontweight="bold", pad=15)

    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Architecture diagram saved to {output_path}")

if __name__ == "__main__":
    create_architecture_diagram()
