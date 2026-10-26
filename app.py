import streamlit as st
import joblib
from pathlib import Path
import matplotlib.pyplot as plt


# Descriptions for predicted news categories
category_descriptions = {
    "business": "Company performance, markets, finance and the economy.",
    "entertainment": "Films, music, television, celebrities and the arts.",
    "politics": "Government decisions, elections, policies and public affairs.",
    "sport": "Sports teams, matches, competitions and athlete performance.",
    "tech": "Technology, software, digital innovation and scientific advances."
}

# -----------------------------
# 1. Page configuration
# -----------------------------
st.set_page_config(
    page_title="NewsLens AI | BBC News Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)
# -----------------------------
# Load trained ML pipeline
# -----------------------------

MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "bbc_news_naive_bayes_pipeline.joblib"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

# Initialize prediction history for the current session
if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []

# -----------------------------
# 2. Custom CSS styling
# -----------------------------
st.markdown(
    """
 <style>
    /* Main application background */
    .stApp {
        background-color: #0b1120;
        color: #e5e7eb;
    }

    /* Sidebar background and readable text */
    [data-testid="stSidebar"] {
        background-color: #111827;
    }

    [data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    /* Main heading */
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #a7b2c5;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    /* Dashboard cards */
    .info-card {
        background-color: #172033;
        border: 1px solid #29364d;
        padding: 1.2rem;
        border-radius: 14px;
        margin-bottom: 1rem;
    }

    .info-label {
        color: #a7b2c5;
        font-size: 0.9rem;
    }

    .info-value {
        color: #f8fafc;
        font-size: 1.5rem;
        font-weight: 700;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #6366f1;
        color: white;
        border: none;
        border-radius: 9px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background-color: #4f46e5;
        color: white;
        border: none;
    }
    
    /* Download button */
    div.stDownloadButton > button {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border: 1px solid #3b82f6 !important;
        border-radius: 9px !important;
        padding: 0.6rem 1.2rem;
        font-weight: 600 !important;
        transition: background-color 0.2s ease;
    }

    /* Download button text */
    div.stDownloadButton > button p,
    div.stDownloadButton > button span {
        color: #ffffff !important;
    }

    /* Download button hover */
    div.stDownloadButton > button:hover {
        background-color: #1d4ed8 !important;
        color: #ffffff !important;
        border: 1px solid #60a5fa !important;
    }

    /* Download button keyboard focus */
    div.stDownloadButton > button:focus {
        color: #ffffff !important;
        outline: 2px solid #60a5fa !important;
        outline-offset: 2px !important;
    }

    /* Text input */
    [data-testid="stTextArea"] textarea {
        background-color: #172033;
        color: #f8fafc;
        border: 1px solid #374151;
        border-radius: 10px;
    }

    [data-testid="stTextArea"] textarea::placeholder {
        color: #9ca3af;
    }

    /* Labels and supporting text */
    [data-testid="stWidgetLabel"] {
        color: #e5e7eb;
    }

    [data-testid="stCaptionContainer"] {
        color: #a7b2c5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# 3. Sidebar
# -----------------------------
with st.sidebar:
    st.title("📰 NewsLens AI")
    st.caption("BBC News Classification")
    st.divider()

    st.subheader("Navigation")
    page = st.radio(
        "Choose a page",
        ["🏠 Home", "🔍 Classify News", "📊 Model Information"],
        label_visibility="collapsed"
    )

    st.divider()
    st.caption("Powered by Machine Learning")
    st.caption("Model: Multinomial Naive Bayes")

# -----------------------------
# 4. Home page
# -----------------------------
if page == "🏠 Home":

    # Hero section
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #172033, #29365a);
            border: 1px solid #374151;
            border-radius: 20px;
            padding: 34px;
            margin-bottom: 28px;
        ">
            <p style="
                color: #a5b4fc;
                font-size: 14px;
                font-weight: 600;
                letter-spacing: 2px;
                margin-bottom: 10px;
            ">MACHINE LEARNING · TEXT CLASSIFICATION</p>
            <h1 style="
                color: #f8fafc;
                font-size: 42px;
                margin: 0 0 14px 0;
            ">Welcome to NewsLens AI</h1>
            <p style="
                color: #cbd5e1;
                font-size: 18px;
                line-height: 1.7;
                max-width: 760px;
                margin-bottom: 0;
            ">
                Turn news text into meaningful categories with a trained
                machine learning pipeline. Explore how AI can understand
                the topics behind an article.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### ✨ What can NewsLens AI do?")

    st.write(
        "Classify a news article into one of five categories, "
        "review estimated category probabilities, and inspect "
        "your previous predictions."
    )

    # Overview metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("News Categories", "05")
        st.caption("Five supported news topics")

    with col2:
        st.metric("Model Algorithm", "Naive Bayes")
        st.caption("Multinomial Naive Bayes")

    with col3:
        st.metric("Input Type", "Text")
        st.caption("News article content")

    # Category cards
    st.markdown("### 🗂️ Explore the News Categories")

    category_cards = [
        (
            "💼",
            "Business",
            "Companies, markets, finance and the economy."
        ),
        (
            "🎬",
            "Entertainment",
            "Films, music, television and the arts."
        ),
        (
            "🏛️",
            "Politics",
            "Government, elections, policies and public affairs."
        ),
        (
            "⚽",
            "Sport",
            "Teams, competitions, matches and sporting news."
        ),
        (
            "💻",
            "Technology",
            "Digital products, computing and technological innovation."
        ),
    ]

    category_columns = st.columns(3)

    for index, (icon, name, description) in enumerate(category_cards):
        with category_columns[index % 3]:
            st.markdown(
                f"""
                <div style="
                    background: #172033;
                    border: 1px solid #303c53;
                    border-radius: 14px;
                    padding: 20px;
                    margin-bottom: 16px;
                    min-height: 150px;
                ">
                    <div style="font-size: 30px; margin-bottom: 10px;">
                        {icon}
                    </div>
                    <h3 style="
                        color: #f8fafc;
                        font-size: 19px;
                        margin: 0 0 8px 0;
                    ">{name}</h3>
                    <p style="
                        color: #a7b2c5;
                        font-size: 14px;
                        line-height: 1.6;
                        margin: 0;
                    ">{description}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Model workflow
    st.markdown("### ⚙️ How it works")

    step1, step2, step3 = st.columns(3)

    with step1:
        st.markdown("**01 · Enter text**")
        st.write("Paste a news article or select a sample article.")

    with step2:
        st.markdown("**02 · Predict category**")
        st.write("The trained pipeline analyzes the text and predicts a category.")

    with step3:
        st.markdown("**03 · Review results**")
        st.write("Explore the predicted category and estimated probabilities.")

    st.info(
        "Ready to try it? Select **Classify News** from the sidebar "
        "to analyze your first article."
    )


# -----------------------------
# 5. Placeholder pages
# -----------------------------

elif page == "🔍 Classify News":
    st.subheader("🔍 Classify a News Article")

    st.write(
        "Paste a news article below and let NewsLens AI "
        "predict its category."
    )

    # Sample articles for quick testing
    sample_articles = {
        "Select a sample article...": "",
        "⚽ Sport": (
            "Manchester United secured an important victory after scoring "
            "two goals in the second half. The manager praised the players "
            "for their performance in the league."
        ),
        "💼 Business": (
            "The company reported higher annual profits as revenue increased. "
            "Investors responded positively to the financial results, "
            "while analysts predicted continued growth."
        ),
        "🏛️ Politics": (
            "The government announced a new policy following a parliamentary "
            "debate. Opposition leaders questioned the proposal, while "
            "ministers defended the planned reforms."
        ),
        "💻 Technology": (
            "Researchers developed a new artificial intelligence system "
            "designed to improve computer performance. Technology companies "
            "are exploring applications in software and cloud computing."
        ),
        "🎬 Entertainment": (
            "The latest film attracted large audiences after its release. "
            "Critics praised the performances, and the director discussed "
            "the production during a television interview."
        )
    }

    # Define the callback BEFORE using it in the selectbox
    def update_sample_article():
        st.session_state["article_input"] = sample_articles[
            st.session_state["sample_selector"]
        ]

    def clear_article():
        st.session_state["article_input"] = ""
        st.session_state["sample_selector"] = "Select a sample article..."

    selected_sample = st.selectbox(
        "Try a sample article",
        options=list(sample_articles.keys()),
        key="sample_selector",
        on_change=update_sample_article
    )

    article_text = st.text_area(
        "News article",
        placeholder="Paste your news article here...",
        height=250,
        help="Enter the text of a news article to classify.",
        key="article_input"
    )

    # Live word counter
    word_count = len(article_text.split()) if article_text.strip() else 0
    st.caption(f"📝 Word count: {word_count}")

    col_classify, col_clear = st.columns([3, 1])

    with col_classify:
        classify_button = st.button(
            "🔍 Classify Article",
            use_container_width=True
        )

    with col_clear:
        st.button(
            "🗑️ Clear Article",
            use_container_width=True,
            on_click=clear_article
        )

    if classify_button:
        if not article_text.strip():
            st.warning("Please enter a news article before classifying.")
        else:
            with st.spinner("Analyzing the article..."):
                prediction = model.predict([article_text])[0]
                probabilities = model.predict_proba([article_text])[0]

            # Article statistics
            word_count = len(article_text.split())
            character_count = len(article_text)
            reading_time = max(1, round(word_count / 200))

            st.markdown("### 📝 Article Statistics")

            stat1, stat2, stat3 = st.columns(3)

            with stat1:
                st.metric("Words", f"{word_count:,}")

            with stat2:
                st.metric("Characters", f"{character_count:,}")

            with stat3:
                st.metric("Reading time", f"{reading_time} min")

            st.success("Classification completed!")

            category_names = {
                "business": "Business",
                "entertainment": "Entertainment",
                "politics": "Politics",
                "sport": "Sport",
                "tech": "Technology"
            }

            category_icons = {
                "business": "💼",
                "entertainment": "🎬",
                "politics": "🏛️",
                "sport": "⚽",
                "tech": "💻"
            }

            display_category = category_names.get(
                prediction, prediction.title()
            )
            category_icon = category_icons.get(prediction, "📰")
            confidence = float(max(probabilities)) * 100

            # Confidence label for the UI
            if confidence >= 80:
                confidence_label = "High estimated probability"
            elif confidence >= 50:
                confidence_label = "Moderate estimated probability"
            else:
                confidence_label = "Low estimated probability"

            # Prediction result card
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #172033, #202b45);
                    border: 1px solid #374151;
                    border-radius: 16px;
                    padding: 26px;
                    margin: 18px 0;
                ">
                    <p style="
                        color: #a7b2c5;
                        font-size: 14px;
                        margin: 0;
                    ">PREDICTED NEWS CATEGORY</p>
                    <h1 style="
                        color: #f8fafc;
                        font-size: 34px;
                        margin: 10px 0;
                    ">{category_icon} {display_category}</h1>
                    <p style="
                        color: #a7b2c5;
                        font-size: 16px;
                        line-height: 1.6;
                        margin-bottom: 0;
                    ">{category_descriptions.get(str(prediction).lower(),
                                                 "A news article classified by the NewsLens AI model.")}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Confidence metric
            metric_col, info_col = st.columns([1, 2])

            with metric_col:
                st.metric(
                    "Top category probability",
                    f"{confidence:.2f}%"
                )

                st.caption(confidence_label)

                # Convert percentage (0–100) to progress (0.0–1.0)
                st.progress(float(confidence) / 100)

            with info_col:
                st.caption(
                    "This is the model's estimated probability for its "
                    "top prediction, not a guarantee that the prediction "
                    "is correct."
                )

            # Probability chart data
            probability_data = {
                category_names.get(label, label.title()): float(prob) * 100
                for label, prob in zip(model.classes_, probabilities)
            }

            # Sort categories from highest to lowest probability
            probability_data = dict(
                sorted(
                    probability_data.items(),
                    key=lambda item: item[1],
                    reverse=True
                )
            )

            st.markdown("### 📊 Category Probability Distribution")

            st.caption(
                "Compare the estimated probability assigned to each category."
            )

            # Create a dark-themed horizontal probability chart
            fig, ax = plt.subplots(figsize=(10, 4))

            # Match the chart and plotting area to the app theme
            fig.patch.set_facecolor("#0b1120")
            ax.set_facecolor("#172033")

            categories = list(probability_data.keys())
            values = list(probability_data.values())

            # Draw horizontal bars
            bars = ax.barh(
                categories,
                values,
                color="#818cf8",
                edgecolor="#a5b4fc",
                height=0.55
            )

            # Show the highest probability at the top
            ax.invert_yaxis()

            # Configure axes and labels
            ax.set_xlabel("Estimated probability (%)",
                          color="#e5e7eb", labelpad=12)
            ax.set_ylabel("News category", color="#e5e7eb", labelpad=12)
            ax.set_xlim(0, 100)

            # Style tick labels
            ax.tick_params(axis="x", colors="#cbd5e1")
            ax.tick_params(axis="y", colors="#e5e7eb")

            # Add subtle grid lines
            ax.set_axisbelow(True)
            ax.grid(axis="x", linestyle="--", alpha=0.2, color="#94a3b8")
            ax.grid(axis="y", visible=False)

            # Remove unnecessary borders
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Add percentage labels to bars
            for bar, value in zip(bars, values):
                if value > 0:
                    label_x = min(value + 1.5, 88)
                    ax.text(
                        label_x,
                        bar.get_y() + bar.get_height() / 2,
                        f"{value:.1f}%",
                        va="center",
                        ha="left",
                        color="#f8fafc",
                        fontsize=10
                    )

            # Improve spacing
            fig.tight_layout()

            # Display in Streamlit and release the figure
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            # Save prediction to session history
            st.session_state["prediction_history"].append({
                "category": display_category,
                "confidence": confidence,
                "words": word_count,
                "article": article_text
            })


elif page == "📊 Model Information":
    st.subheader("📊 Model Information")

    st.write(
        "Explore the architecture, dataset, and workflow behind "
        "the NewsLens AI news classification system."
    )

    # 1. Model overview
    st.markdown("### 🤖 Model Overview")

    model_col1, model_col2, model_col3 = st.columns(3)

    with model_col1:
        st.metric("Algorithm", "Naive Bayes")
        st.caption("Multinomial Naive Bayes classifier")

    with model_col2:
        st.metric("Feature Extraction", "CountVectorizer")
        st.caption("Unigrams and bigrams")

    with model_col3:
        st.metric("Output Categories", "5")
        st.caption("Business, Entertainment, Politics, Sport, Technology")

    st.divider()

    # 2. Dataset overview
    st.markdown("### 📚 Dataset Overview")

    data_col1, data_col2 = st.columns(2)

    with data_col1:
        st.metric("Cleaned Articles", "2,126")
        st.caption("Articles remaining after duplicate removal")

    with data_col2:
        st.metric("News Categories", "5")
        st.caption("Five BBC news topics")

    st.divider()

    # 3. Model architecture
    st.markdown("### ⚙️ Model Architecture")

    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #172033, #202b45);
            border: 1px solid #374151;
            border-radius: 14px;
            padding: 22px;
            text-align: center;
        ">
            <p style="font-size: 18px; color: #f8fafc;">
                📰 <b>News Article Text</b>
            </p>
            <p style="color: #a5b4fc;">↓</p>
            <p style="font-size: 18px; color: #f8fafc;">
                🔢 <b>CountVectorizer</b>
            </p>
            <p style="color: #a5b4fc;">
                Unigrams + Bigrams
            </p>
            <p style="color: #a5b4fc;">↓</p>
            <p style="font-size: 18px; color: #f8fafc;">
                🧠 <b>Multinomial Naive Bayes</b>
            </p>
            <p style="color: #a5b4fc;">↓</p>
            <p style="font-size: 18px; color: #f8fafc;">
                🎯 <b>Predicted News Category</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        "The trained pipeline converts article text into numerical "
        "features before predicting one of the five categories."
    )

    st.divider()

    # 4. Supported categories
    st.markdown("### 🗂️ Supported Categories")

    category_info = {
        "💼 Business": "Companies, markets, finance and the economy.",
        "🎬 Entertainment": "Films, music, television and the arts.",
        "🏛️ Politics": "Government, elections, policies and public affairs.",
        "⚽ Sport": "Teams, competitions, matches and sporting news.",
        "💻 Technology": "Digital products, computing and innovation."
    }

    for category, description in category_info.items():
        with st.expander(category):
            st.write(description)

    st.info(
        "Model probabilities are estimates, not guarantees. "
        "The model is designed for news articles similar to the "
        "BBC news data used during training."
    )

    st.divider()

    # Model evaluation results
    st.markdown("### 📈 Model Evaluation")

    st.write(
        "The following metrics summarize the trained model's "
        "recorded performance on the test dataset."
    )

    eval_col1, eval_col2, eval_col3 = st.columns(3)

    with eval_col1:
        st.metric("Test Accuracy", "97.65%")

    with eval_col2:
        st.metric("Macro F1-score", "97.63%")

    with eval_col3:
        st.metric("Best CV Macro F1", "97.20%")

    st.caption(
        "Cross-validation score: mean macro F1 across 5 folds. "
        "Recorded test results may be optimistic because the test "
        "set was used in earlier model comparisons."
    )

    # Per-category classification report
    st.markdown("#### 🎯 Per-Category Performance")

    import pandas as pd

    report_path = (
        Path(__file__).parent
        / "evaluation"
        / "bbc_news_evaluation_report.csv"
    )

    if report_path.exists():
        report_df = pd.read_csv(report_path, index_col=0)

        # Keep only the five actual news categories
        category_rows = [
            "business",
            "entertainment",
            "politics",
            "sport",
            "tech"
        ]

        category_report = report_df.loc[category_rows].copy()

        # Convert metric scores to percentages
        metric_columns = ["precision", "recall", "f1-score"]
        category_report[metric_columns] = (
            category_report[metric_columns] * 100
        )

        category_report = category_report.rename(
            columns={
                "precision": "Precision (%)",
                "recall": "Recall (%)",
                "f1-score": "F1-score (%)",
                "support": "Test articles"
            }
        )

        category_report.index = [
            "Business",
            "Entertainment",
            "Politics",
            "Sport",
            "Technology"
        ]

        st.dataframe(
            category_report.style.format({
                "Precision (%)": "{:.2f}",
                "Recall (%)": "{:.2f}",
                "F1-score (%)": "{:.2f}",
                "Test articles": "{:.0f}"
            }),
            use_container_width=True
        )

    else:
        st.warning(
            "Evaluation report not found. "
            "Please check the report file location."
        )

    # -----------------------------------------
    # Confusion Matrix
    # -----------------------------------------
    st.divider()

    st.markdown("### 🔍 Confusion Matrix")

    st.markdown(
        """
    Rows represent actual categories; columns represent predicted categories.

    **Diagonal cells** indicate correct predictions. Off-diagonal cells
    indicate misclassified articles.
    """
    )

    confusion_path = (
        Path(__file__).parent
        / "evaluation"
        / "bbc_news_confusion_matrix.csv"
    )

    if confusion_path.exists():
        cm_df = pd.read_csv(confusion_path, index_col=0)

        # Convert labels to readable category names
        readable_names = {
            "business": "Business",
            "entertainment": "Entertainment",
            "politics": "Politics",
            "sport": "Sport",
            "tech": "Technology"
        }

        cm_df.index = [
            readable_names.get(str(label), str(label).title())
            for label in cm_df.index
        ]

        cm_df.columns = [
            readable_names.get(str(label), str(label).title())
            for label in cm_df.columns
        ]

        # Create the heatmap
        fig_cm, ax_cm = plt.subplots(figsize=(9, 6))

        fig_cm.patch.set_facecolor("#0b1120")
        ax_cm.set_facecolor("#172033")

        image_cm = ax_cm.imshow(
            cm_df.values,
            cmap="Blues"
        )

        ax_cm.set_xticks(range(len(cm_df.columns)))
        ax_cm.set_xticklabels(
            cm_df.columns,
            rotation=35,
            ha="right",
            color="#e5e7eb"
        )

        ax_cm.set_yticks(range(len(cm_df.index)))
        ax_cm.set_yticklabels(
            cm_df.index,
            color="#e5e7eb"
        )

        ax_cm.set_xlabel(
            "Predicted Category",
            color="#e5e7eb",
            labelpad=12
        )

        ax_cm.set_ylabel(
            "Actual Category",
            color="#e5e7eb",
            labelpad=12
        )

        # Display the number in every cell
        for row in range(cm_df.shape[0]):
            for col in range(cm_df.shape[1]):
                value = cm_df.iloc[row, col]

                text_color = (
                    "white"
                    if value > cm_df.values.max() * 0.5
                    else "#263445"
                )

                ax_cm.text(
                    col,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=11,
                    fontweight="bold"
                )

        ax_cm.tick_params(axis="both", length=0)

        fig_cm.colorbar(
            image_cm,
            ax=ax_cm,
            fraction=0.046,
            pad=0.04
        )

        fig_cm.tight_layout()

        st.pyplot(
            fig_cm,
            use_container_width=True
        )

        plt.close(fig_cm)

    else:
        st.warning(
            "Confusion matrix file not found. "
            "Check that bbc_news_confusion_matrix.csv "
            "is in the parent folder."
        )

    # -----------------------------------------
    # Key Evaluation Insights
    # -----------------------------------------
    st.divider()

    st.markdown("### 💡 Key Evaluation Insights")

    insight_col1, insight_col2, insight_col3 = st.columns(3)

    with insight_col1:
        st.markdown(
            """
            <div style="
                background: #172033;
                padding: 20px;
                border-radius: 12px;
                border: 1px solid #334155;
            ">
                <p style="color: #cbd5e1; margin: 0;">
                    Correct Predictions
                </p>
                <h2 style="color: #f8fafc; margin: 10px 0;">
                    416 / 426
                </h2>
                <p style="color: #4ade80; margin: 0;">
                    97.65% accuracy
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with insight_col2:
        st.markdown(
            """
            <div style="
                background: #172033;
                padding: 20px;
                border-radius: 12px;
                border: 1px solid #334155;
            ">
                <p style="color: #cbd5e1; margin: 0;">
                    Best Category F1-score
                </p>
                <h2 style="color: #f8fafc; margin: 10px 0;">
                    99.50%
                </h2>
                <p style="color: #4ade80; margin: 0;">
                    Sport
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with insight_col3:
        st.markdown(
            """
            <div style="
                background: #172033;
                padding: 20px;
                border-radius: 12px;
                border: 1px solid #334155;
            ">
                <p style="color: #cbd5e1; margin: 0;">
                    Politics Correctly Classified
                </p>
                <h2 style="color: #f8fafc; margin: 10px 0;">
                    81 / 81
                </h2>
                <p style="color: #4ade80; margin: 0;">
                    100% recall
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("#### 📌 What the Results Tell Us")

    st.markdown(
        """
        - **Strong overall performance:** The model correctly classified
          416 of the 426 test articles.

        - **Excellent sports classification:** Sport achieved an F1-score
          of 99.50%, indicating a strong balance between precision and recall.

        - **Perfect politics recall on this test set:** All 81 actual
          politics articles were correctly identified.

        - **Areas for improvement:** Business and Technology articles
          were sometimes confused with other categories. Further testing
          could help identify ways to reduce these errors.
        """
    )

    st.caption(
        "These insights describe the recorded test-set results. "
        "Performance on new, unseen articles may differ."
    )

    # -----------------------------------------
    # Download Evaluation Report
    # -----------------------------------------
    st.divider()

    st.markdown("### 📥 Download Evaluation Report")

    if report_path.exists():
        report_download_df = pd.read_csv(
            report_path,
            index_col=0
        )

        # Give the category column a proper name
        report_download_df.index.name = "Category"

        # Round evaluation metrics for readability
        metric_columns = [
            "precision",
            "recall",
            "f1-score"
        ]

        report_download_df[metric_columns] = (
            report_download_df[metric_columns].round(4)
        )

        report_download_df["support"] = (
            report_download_df["support"].round(0).astype(int)
        )

        # Correct the accuracy row's support to show total test articles
        if "accuracy" in report_download_df.index:
            report_download_df.loc["accuracy", "support"] = int(
                report_download_df.loc["macro avg", "support"]
            )

        # Export the index as a named Category column
        csv_data = report_download_df.to_csv().encode("utf-8")

        st.download_button(
            label="⬇️ Download Evaluation Report (CSV)",
            data=csv_data,
            file_name="newslens_ai_evaluation_report.csv",
            mime="text/csv",
            use_container_width=True
        )

    else:
        st.warning(
            "Evaluation report is unavailable for download."
        )


# -----------------------------------------
# Prediction History
# -----------------------------------------

if page == "🏠 Home":
    st.markdown("---")
    st.markdown("## 🕘 Prediction History")

    history = st.session_state["prediction_history"]

    if history:
        header_col, button_col = st.columns([3, 1])

        with header_col:
            st.caption(
                f"Total predictions for this session: {len(history)}"
            )

        with button_col:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state["prediction_history"] = []
                st.rerun()

        for index, item in enumerate(reversed(history), start=1):
            with st.expander(
                f"{index}. {item['category']} — "
                f"{item['confidence']:.2f}% estimated probability"
            ):
                detail_col1, detail_col2 = st.columns(2)

                with detail_col1:
                    st.metric(
                        "Estimated probability",
                        f"{item['confidence']:.2f}%"
                    )

                with detail_col2:
                    st.metric("Word count", f"{item['words']:,}")

                st.markdown("**Article text**")
                st.write(item["article"])

    else:
        st.info(
            "Your history is empty. Classify a news article to get started."
        )
