import os
import gradio as gr
import pandas as pd
import numpy as np
import json
import faiss
import matplotlib.pyplot as plt
import pickle
from sentence_transformers import SentenceTransformer
from wordcloud import WordCloud
from utils.utils import download_file_if_missing, download_and_load_csv, check_and_update

# === Download the Data File to the persistent storage ===
# Load version mapping from JSON
with open("version_file_ids.json", "r") as f:
    VERSION_FILE_IDS = json.load(f)
    
# = Load Dataset =
data_file_name = "/data/clinical_trials.csv"
data_file_google_drive_id = "16NWmw2tZ5-5Y-ccIkxr3D81R-LYmjXyV"
print(f"📁 Loading file: {data_file_name}")
check_and_update(data_file_name, data_file_google_drive_id, VERSION_FILE_IDS)
df = pd.read_csv(data_file_name)

# = Download Embeddings =
doc_embeddings_file_name = "/data/doc_embeddings_combined.pkl"
doc_embeddings_google_drive_id = "1i5EZ7sG2J1afmSvXah-9NWu7po6kBGXw"
print(f"📁 Loading file: {doc_embeddings_file_name}")
check_and_update(doc_embeddings_file_name, doc_embeddings_google_drive_id, VERSION_FILE_IDS)

# = Download Vector DB =
vector_db_file_name = "/data/faiss_index.bin"
vector_db_google_drive_id = "1VVIqNnFaG3_wKfXC8xBg0mEL7kZWeDk_"
print(f"📁 Loading file: {vector_db_file_name}")
check_and_update(vector_db_file_name, vector_db_google_drive_id, VERSION_FILE_IDS)

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
def search_trials(query_text, num_to_fetch=5):
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
        study_type = df.iloc[idx]["Study Type"]
        study_design = df.iloc[idx]["Study Design"]
        study_status = df.iloc[idx]["Study Status"]
        age = df.iloc[idx]["Age"]
        gender = df.iloc[idx]["Sex"]
        enrollments = df.iloc[idx]["Enrollment"]
        raw_text = X_train_text[idx]
        img_path = generate_wordcloud_image(raw_text, idx)
        results.append((img_path,
                        f"Clinical Trial ID: {ct_id}",
                        study_title,
                        brief_summary,
                        conditions,
                        interventions,
                        primary_outcome_measures,
                        study_type,
                        study_design,
                        study_status,
                        age,
                        gender,
                        enrollments
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
                with gr.Row():
                    image1 = gr.Image(label="1")
                with gr.Row():
                    txt_study_type1 = gr.Textbox(lines=1, label="Study Type")
                    txt_study_design1 = gr.Textbox(lines=1, label="Study Design")
                    txt_study_status1 = gr.Textbox(lines=1, label="Study Status")
                with gr.Row():
                    txt_age1 = gr.Textbox(lines=1, label="Age")
                    txt_gender1 = gr.Textbox(lines=1, label="Gender")
                    txt_enrollments1 = gr.Textbox(lines=1, label="Enrollments")

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
                with gr.Row():
                    image2 = gr.Image(label="2")
                with gr.Row():
                    txt_study_type2 = gr.Textbox(lines=1, label="Study Type")
                    txt_study_design2 = gr.Textbox(lines=1, label="Study Design")
                    txt_study_status2 = gr.Textbox(lines=1, label="Study Status")
                with gr.Row():
                    txt_age2 = gr.Textbox(lines=1, label="Age")
                    txt_gender2 = gr.Textbox(lines=1, label="Gender")
                    txt_enrollments2 = gr.Textbox(lines=1, label="Enrollments")

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
                with gr.Row():
                    image3 = gr.Image(label="3")
                with gr.Row():
                    txt_study_type3 = gr.Textbox(lines=1, label="Study Type")
                    txt_study_design3 = gr.Textbox(lines=1, label="Study Design")
                    txt_study_status3 = gr.Textbox(lines=1, label="Study Status")
                with gr.Row():
                    txt_age3 = gr.Textbox(lines=1, label="Age")
                    txt_gender3 = gr.Textbox(lines=1, label="Gender")
                    txt_enrollments3 = gr.Textbox(lines=1, label="Enrollments")

    with gr.Accordion("Semantically Similar Clinical Trial #4"):
        html_study_title4 = gr.HTML()
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    txt_brief_summary4 = gr.Textbox(lines=2, max_length=5, label="Brief Summary")
                with gr.Row():
                        txt_conditions4 = gr.Textbox(lines=1, label="Conditions")
                        txt_interventions4 = gr.Textbox(lines=1, label="Interventions")
                with gr.Row():
                    txt_primary_outcome4 = gr.Textbox(lines=3, max_length=5, label="Primary Outcome Measures")
            with gr.Column():
                with gr.Row():
                    image4 = gr.Image(label="4")
                with gr.Row():
                    txt_study_type4 = gr.Textbox(lines=1, label="Study Type")
                    txt_study_design4 = gr.Textbox(lines=1, label="Study Design")
                    txt_study_status4 = gr.Textbox(lines=1, label="Study Status")
                with gr.Row():
                    txt_age4 = gr.Textbox(lines=1, label="Age")
                    txt_gender4 = gr.Textbox(lines=1, label="Gender")
                    txt_enrollments4 = gr.Textbox(lines=1, label="Enrollments")

    with gr.Accordion("Semantically Similar Clinical Trial #5"):
        html_study_title5 = gr.HTML()
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    txt_brief_summary5 = gr.Textbox(lines=2, max_length=5, label="Brief Summary")
                with gr.Row():
                        txt_conditions5 = gr.Textbox(lines=1, label="Conditions")
                        txt_interventions5 = gr.Textbox(lines=1, label="Interventions")
                with gr.Row():
                    txt_primary_outcome5 = gr.Textbox(lines=3, max_length=5, label="Primary Outcome Measures")
            with gr.Column():
                with gr.Row():
                    image5 = gr.Image(label="5")
                with gr.Row():
                    txt_study_type5 = gr.Textbox(lines=1, label="Study Type")
                    txt_study_design5 = gr.Textbox(lines=1, label="Study Design")
                    txt_study_status5 = gr.Textbox(lines=1, label="Study Status")
                with gr.Row():
                    txt_age5 = gr.Textbox(lines=1, label="Age")
                    txt_gender5 = gr.Textbox(lines=1, label="Gender")
                    txt_enrollments5 = gr.Textbox(lines=1, label="Enrollments")

    def run_search(q):
        results = search_trials(q)
        image_outputs = []
        brief_summary_outputs = []
        conditions_outputs = []
        interventions_outputs = []
        primary_outcomes_outputs = []
        trial_title = []
        study_type_outputs = []
        study_design_outputs = []
        study_status_outputs = []
        age_outputs = []
        gender_outputs = []
        enrollments_outputs = []

        for i, (img_url, 
                label, 
                study_title, 
                brief_summary,
                conditions,
                interventions,
                primary_outcomes,
                study_type,
                study_design,
                study_status,
                age,
                gender,
                enrollments
               ) in enumerate(results):
            image_outputs.append(img_url)
            brief_summary_outputs.append(brief_summary)
            conditions_outputs.append(conditions)
            interventions_outputs.append(interventions)
            primary_outcomes_outputs.append(primary_outcomes)
            study_type_outputs.append(study_type)
            study_design_outputs.append(study_design)
            study_status_outputs.append(study_status)
            age_outputs.append(age)
            gender_outputs.append(gender)
            enrollments_outputs.append(enrollments)

            trial_title.append(f"""
                                    <div style='text-align: center; margin-bottom: 10px;'>
                                        <p style='font-size: 20px; font-weight: bold;'>{label}</p>
                                        <p style='font-size: 16px; line-height: 1.6; max-width: 900px; margin: auto;'>
                                            Study Title: <b>{study_title}</b>
                                        </p>
                                    </div>
                                """
                              )

        return image_outputs[0], trial_title[0], brief_summary_outputs[0], conditions_outputs[0], interventions_outputs[0], primary_outcomes_outputs[0], study_type_outputs[0], study_design_outputs[0], study_status_outputs[0], age_outputs[0], gender_outputs[0], enrollments_outputs[0], \
               image_outputs[1], trial_title[1], brief_summary_outputs[1], conditions_outputs[1], interventions_outputs[1], primary_outcomes_outputs[1], study_type_outputs[1], study_design_outputs[1], study_status_outputs[1], age_outputs[1], gender_outputs[1], enrollments_outputs[1], \
               image_outputs[2], trial_title[2], brief_summary_outputs[2], conditions_outputs[2], interventions_outputs[2], primary_outcomes_outputs[2], study_type_outputs[2], study_design_outputs[2], study_status_outputs[2], age_outputs[2], gender_outputs[2], enrollments_outputs[2], \
               image_outputs[3], trial_title[3], brief_summary_outputs[3], conditions_outputs[3], interventions_outputs[3], primary_outcomes_outputs[3], study_type_outputs[3], study_design_outputs[3], study_status_outputs[3], age_outputs[3], gender_outputs[3], enrollments_outputs[3], \
               image_outputs[4], trial_title[4], brief_summary_outputs[4], conditions_outputs[4], interventions_outputs[4], primary_outcomes_outputs[4], study_type_outputs[4], study_design_outputs[4], study_status_outputs[4], age_outputs[4], gender_outputs[4], enrollments_outputs[4]

    submit_btn.click(
        fn=run_search, 
        inputs=query_input, 
        outputs=[image1, html_study_title1, txt_brief_summary1, txt_conditions1, txt_interventions1, txt_primary_outcome1, txt_study_type1, txt_study_design1, txt_study_status1, txt_age1, txt_gender1, txt_enrollments1,
                 image2, html_study_title2, txt_brief_summary2, txt_conditions2, txt_interventions2, txt_primary_outcome2, txt_study_type2, txt_study_design2, txt_study_status2, txt_age2, txt_gender2, txt_enrollments2,
                 image3, html_study_title3, txt_brief_summary3, txt_conditions3, txt_interventions3, txt_primary_outcome3, txt_study_type3, txt_study_design3, txt_study_status3, txt_age3, txt_gender3, txt_enrollments3,
                 image4, html_study_title4, txt_brief_summary4, txt_conditions4, txt_interventions4, txt_primary_outcome4, txt_study_type4, txt_study_design4, txt_study_status4, txt_age4, txt_gender4, txt_enrollments4,
                 image5, html_study_title5, txt_brief_summary5, txt_conditions5, txt_interventions5, txt_primary_outcome5, txt_study_type5, txt_study_design5, txt_study_status5, txt_age5, txt_gender5, txt_enrollments5
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
    on_spaces = os.getenv("SPACE_ID") is not None
    port = int(os.getenv("PORT", 7860))  # HF sets PORT

    app.queue().launch(
        server_name="0.0.0.0",   # listen on all interfaces
        server_port=port,        # use HF-provided port
        show_api=False,
        share=not on_spaces      # False on Spaces, True locally
    )