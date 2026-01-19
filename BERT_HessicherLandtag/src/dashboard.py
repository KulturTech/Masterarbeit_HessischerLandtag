import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="BERT Classification Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🤖 BERT Hate Speech Classification Dashboard</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar - File Upload & Filters
with st.sidebar:
    st.header("⚙️ Configuration")

    # File upload
    data_path = st.text_input(
        "Data Path",
        value=r"c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\prep_v1\all_docs_classified.parquet",
        help="Path to the classified parquet file"
    )

    if st.button("🔄 Load Data", type="primary"):
        st.session_state.reload = True

    st.markdown("---")

    # Filters section
    st.header("🔍 Filters")

# Load data
@st.cache_data
def load_data(path):
    try:
        df = pd.read_parquet(path)
        df['text_length'] = df['text'].str.len()
        df['word_count'] = df['text'].str.split().str.len()
        return df
    except FileNotFoundError:
        return None

# Load the data
df = load_data(data_path)

if df is None:
    st.error(f"❌ Could not find data at: {data_path}")
    st.info("Please run the classification script first: `python src/test.py`")
    st.stop()

# Sidebar filters
with st.sidebar:
    # Label filter
    all_labels = ['All'] + sorted(df['label'].unique().tolist())
    selected_label = st.selectbox("Label", all_labels)

    # Confidence filter
    min_confidence, max_confidence = st.slider(
        "Confidence Score Range",
        min_value=0.0,
        max_value=1.0,
        value=(0.0, 1.0),
        step=0.05
    )

    # Text length filter
    min_length, max_length = st.slider(
        "Text Length Range (characters)",
        min_value=int(df['text_length'].min()),
        max_value=int(df['text_length'].max()),
        value=(int(df['text_length'].min()), int(df['text_length'].max())),
        step=100
    )

    st.markdown("---")

    # Export options
    st.header("📥 Export")
    if st.button("Download Filtered Data"):
        filtered_df = df.copy()
        if selected_label != 'All':
            filtered_df = filtered_df[filtered_df['label'] == selected_label]
        filtered_df = filtered_df[
            (filtered_df['score'] >= min_confidence) &
            (filtered_df['score'] <= max_confidence) &
            (filtered_df['text_length'] >= min_length) &
            (filtered_df['text_length'] <= max_length)
        ]
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="💾 Download CSV",
            data=csv,
            file_name="filtered_classifications.csv",
            mime="text/csv"
        )

# Apply filters
filtered_df = df.copy()
if selected_label != 'All':
    filtered_df = filtered_df[filtered_df['label'] == selected_label]
filtered_df = filtered_df[
    (filtered_df['score'] >= min_confidence) &
    (filtered_df['score'] <= max_confidence) &
    (filtered_df['text_length'] >= min_length) &
    (filtered_df['text_length'] <= max_length)
]

# Overview metrics
st.header("📊 Overview")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Documents", f"{len(df):,}")
with col2:
    st.metric("Filtered Documents", f"{len(filtered_df):,}")
with col3:
    st.metric("Avg Confidence", f"{filtered_df['score'].mean():.3f}")
with col4:
    st.metric("Unique Labels", len(df['label'].unique()))
with col5:
    st.metric("Low Confidence (<0.6)", f"{len(filtered_df[filtered_df['score'] < 0.6]):,}")

st.markdown("---")

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Label Distribution",
    "🎯 Confidence Analysis",
    "📝 Text Analysis",
    "🔎 Document Explorer",
    "📊 Advanced Analytics",
    "🚨 Hate Speech Against Immigrants"
])

# TAB 1: Label Distribution
with tab1:
    st.header("Label Distribution")

    col1, col2 = st.columns(2)

    with col1:
        # Bar chart
        label_counts = filtered_df['label'].value_counts().reset_index()
        label_counts.columns = ['Label', 'Count']
        label_counts['Percentage'] = (label_counts['Count'] / len(filtered_df) * 100).round(2)

        fig = px.bar(
            label_counts,
            x='Label',
            y='Count',
            text='Count',
            title="Document Count by Label",
            color='Count',
            color_continuous_scale='Blues'
        )
        fig.update_traces(texttemplate='%{text}<br>(%{customdata[0]:.1f}%)',
                         textposition='outside',
                         customdata=label_counts[['Percentage']])
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Pie chart
        fig = px.pie(
            label_counts,
            values='Count',
            names='Label',
            title="Label Distribution (%)",
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Data table
    st.subheader("Label Statistics")
    st.dataframe(
        label_counts.style.format({'Percentage': '{:.2f}%'}),
        use_container_width=True,
        hide_index=True
    )

# TAB 2: Confidence Analysis
with tab2:
    st.header("Confidence Score Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Overall histogram
        fig = px.histogram(
            filtered_df,
            x='score',
            nbins=50,
            title="Overall Confidence Score Distribution",
            labels={'score': 'Confidence Score', 'count': 'Frequency'},
            color_discrete_sequence=['steelblue']
        )
        fig.add_vline(x=filtered_df['score'].mean(), line_dash="dash",
                     line_color="red", annotation_text="Mean")
        fig.add_vline(x=filtered_df['score'].median(), line_dash="dash",
                     line_color="green", annotation_text="Median")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Box plot by label
        fig = px.box(
            filtered_df,
            x='label',
            y='score',
            title="Confidence Score Distribution by Label",
            color='label',
            labels={'score': 'Confidence Score', 'label': 'Label'}
        )
        st.plotly_chart(fig, use_container_width=True)

    # Confidence level breakdown
    st.subheader("Confidence Levels")
    filtered_df['confidence_level'] = pd.cut(
        filtered_df['score'],
        bins=[0, 0.6, 0.8, 1.0],
        labels=['Low (<0.6)', 'Medium (0.6-0.8)', 'High (>0.8)']
    )

    conf_counts = filtered_df['confidence_level'].value_counts().reset_index()
    conf_counts.columns = ['Confidence Level', 'Count']
    conf_counts['Percentage'] = (conf_counts['Count'] / len(filtered_df) * 100).round(2)

    fig = px.bar(
        conf_counts,
        x='Confidence Level',
        y='Count',
        text='Count',
        title="Documents by Confidence Level",
        color='Confidence Level',
        color_discrete_map={'Low (<0.6)': '#d32f2f',
                           'Medium (0.6-0.8)': '#ff9800',
                           'High (>0.8)': '#388e3c'}
    )
    fig.update_traces(texttemplate='%{text}<br>(%{customdata[0]:.1f}%)',
                     customdata=conf_counts[['Percentage']])
    st.plotly_chart(fig, use_container_width=True)

# TAB 3: Text Analysis
with tab3:
    st.header("Text Length Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Character length distribution
        fig = px.histogram(
            filtered_df,
            x='text_length',
            nbins=50,
            title="Text Length Distribution (Characters)",
            labels={'text_length': 'Text Length', 'count': 'Frequency'}
        )
        fig.add_vline(x=filtered_df['text_length'].mean(), line_dash="dash",
                     line_color="red", annotation_text=f"Mean: {filtered_df['text_length'].mean():.0f}")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Word count distribution
        fig = px.histogram(
            filtered_df,
            x='word_count',
            nbins=50,
            title="Word Count Distribution",
            labels={'word_count': 'Word Count', 'count': 'Frequency'},
            color_discrete_sequence=['coral']
        )
        fig.add_vline(x=filtered_df['word_count'].mean(), line_dash="dash",
                     line_color="red", annotation_text=f"Mean: {filtered_df['word_count'].mean():.0f}")
        st.plotly_chart(fig, use_container_width=True)

    # Scatter plots
    st.subheader("Text Length vs Confidence")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(
            filtered_df.sample(min(1000, len(filtered_df))),
            x='text_length',
            y='score',
            color='label',
            title="Text Length vs Confidence Score",
            labels={'text_length': 'Text Length (chars)', 'score': 'Confidence'},
            opacity=0.6
        )
        corr = filtered_df['text_length'].corr(filtered_df['score'])
        fig.add_annotation(text=f"Correlation: {corr:.3f}",
                          xref="paper", yref="paper",
                          x=0.05, y=0.95, showarrow=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(
            filtered_df.sample(min(1000, len(filtered_df))),
            x='word_count',
            y='score',
            color='label',
            title="Word Count vs Confidence Score",
            labels={'word_count': 'Word Count', 'score': 'Confidence'},
            opacity=0.6
        )
        corr = filtered_df['word_count'].corr(filtered_df['score'])
        fig.add_annotation(text=f"Correlation: {corr:.3f}",
                          xref="paper", yref="paper",
                          x=0.05, y=0.95, showarrow=False)
        st.plotly_chart(fig, use_container_width=True)

# TAB 4: Document Explorer
with tab4:
    st.header("Document Explorer")

    # Search functionality
    search_query = st.text_input("🔍 Search documents", "")

    # Sort options
    col1, col2, col3 = st.columns(3)
    with col1:
        sort_by = st.selectbox("Sort by", ['score', 'text_length', 'word_count'])
    with col2:
        sort_order = st.radio("Order", ['Ascending', 'Descending'], horizontal=True)
    with col3:
        show_n = st.number_input("Show top N", min_value=10, max_value=1000, value=50, step=10)

    # Apply search and sort
    display_df = filtered_df.copy()
    if search_query:
        display_df = display_df[display_df['text'].str.contains(search_query, case=False, na=False)]

    display_df = display_df.sort_values(by=sort_by, ascending=(sort_order == 'Ascending'))
    display_df = display_df.head(show_n)

    st.info(f"Showing {len(display_df)} documents")

    # Display documents
    for idx, row in display_df.iterrows():
        with st.expander(f"📄 Document {row.get('doc_id', idx)} | Label: {row['label']} | Confidence: {row['score']:.3f}"):
            st.markdown(f"**Text:** {row['text'][:500]}{'...' if len(row['text']) > 500 else ''}")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence", f"{row['score']:.3f}")
            with col2:
                st.metric("Text Length", f"{row['text_length']} chars")
            with col3:
                st.metric("Word Count", f"{row['word_count']} words")

# TAB 5: Advanced Analytics
with tab5:
    st.header("Advanced Analytics")

    # Heatmap: Label vs Confidence Level
    st.subheader("Label vs Confidence Level Heatmap")

    heatmap_data = pd.crosstab(
        filtered_df['label'],
        filtered_df['confidence_level']
    )

    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Confidence Level", y="Label", color="Count"),
        title="Distribution of Labels by Confidence Level",
        color_continuous_scale='YlOrRd',
        text_auto=True
    )
    st.plotly_chart(fig, use_container_width=True)

    # Per-label statistics
    st.subheader("Detailed Label Statistics")

    stats_df = filtered_df.groupby('label').agg({
        'score': ['count', 'mean', 'median', 'std', 'min', 'max'],
        'text_length': ['mean', 'median'],
        'word_count': ['mean', 'median']
    }).round(3)

    stats_df.columns = ['_'.join(col).strip() for col in stats_df.columns.values]
    stats_df = stats_df.reset_index()

    st.dataframe(stats_df, use_container_width=True)

    # 3D Scatter plot
    st.subheader("3D Analysis: Text Length vs Word Count vs Confidence")

    sample_df = filtered_df.sample(min(500, len(filtered_df)))

    fig = px.scatter_3d(
        sample_df,
        x='text_length',
        y='word_count',
        z='score',
        color='label',
        title="3D Visualization",
        labels={
            'text_length': 'Text Length',
            'word_count': 'Word Count',
            'score': 'Confidence'
        },
        opacity=0.7
    )
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 1rem;'>
    <p>🤖 BERT Hate Speech Classification Dashboard | Built with Streamlit & Plotly</p>
</div>
""", unsafe_allow_html=True)
