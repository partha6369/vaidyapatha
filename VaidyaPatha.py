import os
import gradio as gr
import pandas as pd
import numpy as np
import faiss
import matplotlib.pyplot as plt
import pickle
from sentence_transformers import SentenceTransformer
from wordcloud import WordCloud
from utils.utils import download_file_if_missing, download_and_load_csv

# === Download the Data File to the persistent storage ===
# = Load Dataset =
data_file_name = "/data/clinical_trials.csv"
data_file_google_drive_id = "16NWmw2tZ5-5Y-ccIkxr3D81R-LYmjXyV"
df = download_and_load_csv(data_file_name, data_file_google_drive_id)

# = Download Embeddings =
doc_embeddings_file_name = "/data/doc_embeddings_combined.pkl"
doc_embeddings_google_drive_id = "1i5EZ7sG2J1afmSvXah-9NWu7po6kBGXw"
download_file_if_missing(doc_embeddings_file_name, doc_embeddings_google_drive_id)

# = Download Vector DB =
vector_db_file_name = "/data/faiss_index.bin"
vector_db_google_drive_id = "1VVIqNnFaG3_wKfXC8xBg0mEL7kZWeDk_"
download_file_if_missing(vector_db_file_name, vector_db_google_drive_id)

# === Prepare combined_text list ===
X_train_text = df["combined_text"].fillna("").tolist()

# === Load Embeddings ===
with open(doc_embeddings_file_name, "rb") as f:
    doc_embeddings = pickle.load(f)
doc_embeddings_np = np.array(doc_embeddings).astype("float32")

# === Load FAISS Index ===
index = faiss.read_index(vector_db_file_name)

# === Load Embedding Model ===
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# === Helper Function to Generate Word Cloud Image ===
def generate_wordcloud_image(text, idx):
    output_dir = "wordclouds"
    os.makedirs(output_dir, exist_ok=True)  # Create folder if it doesn't exist
    filename = os.path.join(output_dir, f"wordcloud_{idx}.png")

    if os.path.exists(filename):
        return filename

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    wordcloud.to_file(filename)
    return filename

# === Gradio Inference Function ===
def search_trials(query_text, num_to_fetch=3):
    if not query_text.strip():
        return [(None, "Please enter a valid query.", "")]

    query_embedding = embed_model.encode([query_text], normalize_embeddings=True).astype("float32")
    D, I = index.search(query_embedding, num_to_fetch)

    results = []
    for idx in I[0]:
        ct_id = df.iloc[idx]["CT_ID"]
        study_title = df.iloc[idx]["Study Title_original"]
        brief_summary = df.iloc[idx]["Brief Summary_original"]
        conditions = df.iloc[idx]["Conditions_original"]
        interventions = df.iloc[idx]["Interventions_original"]
        primary_outcome_measures = df.iloc[idx]["Primary Outcome Measures_original"]
        raw_text = X_train_text[idx]
        img_path = generate_wordcloud_image(raw_text, idx)
        results.append((img_path,
                        f"Clinical Trial ID: {ct_id}",
                        study_title,
                        brief_summary,
                        conditions,
                        interventions,
                        primary_outcome_measures
                       )
                      )

    return results

# Try to load the PayPal URL from the environment; if missing, use a placeholder
paypal_url = os.getenv("PAYPAL_URL", "https://www.paypal.com/donate/dummy-link")

APP_TITLE = "🧠 VaidyaPatha (वैद्यपथ) — Semantic Clinical Trial Explorer"
APP_DESCRIPTION = (
    "Enter a disease, condition, or treatment to find the most relevant clinical trials and their summaries."
)

# === Gradio Interface ===
with gr.Blocks(title="VaidyaPatha: Clinical Trial Semantic Search") as app:
    gr.HTML(f"""
        <div style='text-align: center; margin-bottom: 30px;'>
            <p style='font-size: 40px; font-weight: bold;'>{APP_TITLE}</p>
            <p style='font-size: 20px; line-height: 1.6; max-width: 900px; margin: auto;'>
                {APP_DESCRIPTION}
            </p>
        </div>
    """)

    with gr.Row():
        query_input = gr.Textbox(label="Enter Query", placeholder="e.g., brain tumour clinical trials")
        submit_btn = gr.Button("🔍 Search")

    # Individual output components
    with gr.Accordion("Semantically Similar Clinical Trial #1"):
        html_study_title1 = gr.HTML()
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    txt_brief_summary1 = gr.Textbox(lines=2, max_length=5, label="Brief Summary")
                with gr.Row():
                        txt_conditions1 = gr.Textbox(lines=1, label="Conditions")
                        txt_interventions1 = gr.Textbox(lines=1, label="Interventions")
                with gr.Row():
                    txt_primary_outcome1 = gr.Textbox(lines=3, max_length=5, label="Primary Outcome Measures")
            with gr.Column():
                image1 = gr.Image(label="1")

    with gr.Accordion("Semantically Similar Clinical Trial #2"):
        html_study_title2 = gr.HTML()
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    txt_brief_summary2 = gr.Textbox(lines=2, max_length=5, label="Brief Summary")
                with gr.Row():
                        txt_conditions2 = gr.Textbox(lines=1, label="Conditions")
                        txt_interventions2 = gr.Textbox(lines=1, label="Interventions")
                with gr.Row():
                    txt_primary_outcome2 = gr.Textbox(lines=3, max_length=5, label="Primary Outcome Measures")
            with gr.Column():
                image2 = gr.Image(label="2")

    with gr.Accordion("Semantically Similar Clinical Trial #3"):
        html_study_title3 = gr.HTML()
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    txt_brief_summary3 = gr.Textbox(lines=2, max_length=5, label="Brief Summary")
                with gr.Row():
                        txt_conditions3 = gr.Textbox(lines=1, label="Conditions")
                        txt_interventions3 = gr.Textbox(lines=1, label="Interventions")
                with gr.Row():
                    txt_primary_outcome3 = gr.Textbox(lines=3, max_length=5, label="Primary Outcome Measures")
            with gr.Column():
                image3 = gr.Image(label="3")

    def run_search(q):
        results = search_trials(q)
        image_outputs = []
        brief_summary_outputs = []
        conditions_outputs = []
        interventions_outputs = []
        primary_outcomes_outputs = []
        trial_title = []

        for i, (img_url, 
                label, 
                study_title, 
                brief_summary,
                conditions,
                interventions,
                primary_outcomes
               ) in enumerate(results):
            image_outputs.append(img_url)
            brief_summary_outputs.append(brief_summary)
            conditions_outputs.append(conditions)
            interventions_outputs.append(interventions)
            primary_outcomes_outputs.append(primary_outcomes)

            trial_title.append(f"""
                                    <div style='text-align: center; margin-bottom: 10px;'>
                                        <p style='font-size: 20px; font-weight: bold;'>{label}</p>
                                        <p style='font-size: 16px; line-height: 1.6; max-width: 900px; margin: auto;'>
                                            Study Title: <b>{study_title}</b>
                                        </p>
                                    </div>
                                """
                              )

        return image_outputs[0], trial_title[0], brief_summary_outputs[0], conditions_outputs[0], interventions_outputs[0], primary_outcomes_outputs[0], \
               image_outputs[1], trial_title[1], brief_summary_outputs[1], conditions_outputs[1], interventions_outputs[1], primary_outcomes_outputs[1], \
               image_outputs[2], trial_title[2], brief_summary_outputs[2], conditions_outputs[2], interventions_outputs[2], primary_outcomes_outputs[2]

    submit_btn.click(
        fn=run_search, 
        inputs=query_input, 
        outputs=[image1, html_study_title1, txt_brief_summary1, txt_conditions1, txt_interventions1, txt_primary_outcome1,
                 image2, html_study_title2, txt_brief_summary2, txt_conditions2, txt_interventions2, txt_primary_outcome2,
                 image3, html_study_title3, txt_brief_summary3, txt_conditions3, txt_interventions3, txt_primary_outcome3
                ]
    )
    
    gr.HTML(f"""
        <a href="{paypal_url}" target="_blank">
            <button style="background-color:#0070ba;color:white;border:none;padding:10px 20px;
            font-size:16px;border-radius:5px;cursor:pointer;margin-top:10px;">
                ❤️ Support Research via PayPal
            </button>
        </a>
        """)

# === Launch the App ===
if __name__ == "__main__":
    # Determine if running on Hugging Face Spaces
    on_spaces = os.environ.get("SPACE_ID") is not None
    
    # Launch the app conditionally
    app.launch(share=not on_spaces)