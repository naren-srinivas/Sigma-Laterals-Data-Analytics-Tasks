import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import time

# Set page configuration and styling
st.set_page_config(
    page_title="Customer Segmentation Analysis", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Header styling */
    .main-header {
        background-color: #4267B2;
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .dashboard-card {
        background-color: black;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    /* Tab styling for horizontal layout with even spacing */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        border-radius: 10px;
        padding: 5px;
        display: flex;
        justify-content: space-evenly;
        width: 100%;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1d1e1f;
        padding-right: 2px;
        padding-left: 2px;
        border: 5px;
        border-radius: 5px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        flex: 1;
        text-align: center;
    }
        
    .stTabs [aria-selected="true"] {
        background-color: #4267B2;
        color: white;
    }
    
    /* File uploader styling */
    .uploadedFile {
        border-radius: 10px;
        padding: 5px;
    }
    
    /* Fix chart width */
    .stPlotlyChart {
        width: 100%;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        background-color: #f1f3f4;
        border-radius: 5px;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #4267B2;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
    }
    
    .stButton>button:hover {
        background-color: #365899;
    }
    
    /* Success messages */
    .success-msg {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Metric cards */
    .metric-card {
        background-color: white;
        border-radius: 8px;
        margin:3px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        padding: 1rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #4267B2;
    }
    
    .metric-label {
        font-size: 14px;
        color: #6c757d;
    }
    
    /* Multiselect styling */
    div[data-baseweb="select"] {
        border-radius: 5px;
    }
    
    /* Minimize the file uploader when files are uploaded */
    .minimized-uploader {
        padding: 10px;
        background-color: #F44336;
        border-radius: 5px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

class CustomerSegmentation:
    def __init__(self, df):
        
        #client = OpenAI(api_key = )
        
        self.summary_columns = ["Age", "Tenure", "Income", "TotalSpend", "SpendPerChild",
            "TotalPurchases", "PurchaseDiversity", "OnlinePurchaseRatio",
            "Recency", "DiscountAffinity", "Has_Kids", "CampaignEngaged",
            "EngagedOnline", "ChurnRisk"]
        self.summary = None
        
        df, columns_normalize, X_scaled = self.feature_engineer(self.load_data(df))
        self.df = df
        self.columns_normalize = columns_normalize
        self.X_scaled = X_scaled
        
        
    def load_data(self, df):
        education_order = {"Basic": 0, "2n Cycle": 1, "Graduation": 2, "Master": 3, "PhD": 4}
        df["Education"] = df["Education"].map(education_order).fillna(-1)
        
        df["Marital_Status"] = df["Marital_Status"].replace({"Alone": "Single", "Absurd": "Single", "YOLO": "Single", "Together": "Married"})
        df = pd.get_dummies(df, dtype=int, columns=["Marital_Status"], drop_first=False)# Using drop_first from https://medium.com/@randhirsingh23/why-use-get-dummies-with-drop-first-in-machine-learning-16be85d3899c
        
        x = df["Income"].median()
        df.fillna({"Income": x}, inplace=True)
        df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y")
        
        return df

    def feature_engineer(self, df):
        df["Age"] = datetime.now().year - df["Year_Birth"]
        df["Tenure"] = (datetime.now() - df["Dt_Customer"]).dt.days // 30  # It is in months
        
        df["Children"] = df["Kidhome"] + df["Teenhome"]
        df["Has_Kids"] = (df["Children"] > 0).astype(int)
        
        mntcolumns = [col for col in df.columns if col.startswith("Mnt")]
        df["TotalSpend"] = df[mntcolumns].sum(axis=1)  # Sum taken along a row
        
        for col in mntcolumns:
            df[col + "Percent"] = df[col] / df["TotalSpend"]
            df[col + "Percent"] = df[col + "Percent"].fillna(0)  # if TotalSpend = 0
        df["SpendPerChild"] = df["TotalSpend"] / (df["Children"] + 1)
        
        df["TotalPurchases"] = df["NumWebPurchases"] + df["NumCatalogPurchases"] + df["NumStorePurchases"]
        df["PurchaseDiversity"] = ((df[["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]] > 0).sum(axis=1))
        
        df["OnlinePurchaseRatio"] = df["NumWebPurchases"] / df["TotalPurchases"]
        df["OnlinePurchaseRatio"] = df["OnlinePurchaseRatio"].fillna(0)
        online_visits_threshold = 5  # example threshold
        df["EngagedOnline"] = (df["NumWebVisitsMonth"] > online_visits_threshold).astype(int)
        
        cmp_cols = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Response"]
        df["TotalAccepted"] = df[cmp_cols].sum(axis=1)
        df["CampaignEngaged"] = (df["TotalAccepted"] > 0).astype(int)
        df["DiscountAffinity"] = df["NumDealsPurchases"] / df["TotalPurchases"]
        df["DiscountAffinity"] = df["DiscountAffinity"].fillna(0)
        
        df["ComplainedHighValue"] = (df["Complain"] == 1) & (df["TotalSpend"] > df["TotalSpend"].median())
        
        recency_threshold = df["Recency"].median()
        df["ChurnRisk"] = ((df["Recency"] > recency_threshold) & (df["Complain"] == 1)).astype(int)
        
        drop_columns = ["Dt_Customer", "Year_Birth", "Kidhome", "Teenhome", 
                        "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", 
                        "AcceptedCmp5", "Response", "Complain"]
        df.drop(columns=drop_columns, inplace=True)

        columns_normalize = ["Age", "Tenure", "Income", "TotalSpend", "SpendPerChild", 
                            "TotalPurchases", "PurchaseDiversity", "OnlinePurchaseRatio", 
                            "Recency", "Z_CostContact", "Z_Revenue"] + \
                            [col for col in df.columns if col.startswith("Mnt") and not col.endswith("Percent")]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[columns_normalize])
    
        return df, columns_normalize, X_scaled
        
    def find_optimal_k(self): # elboqw method
        wcss = []
        silhouette_scores = []
        K_range = range(2, 11)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.X_scaled)
            wcss.append(kmeans.inertia_)
            
            labels = kmeans.labels_
            silhouette_scores.append(silhouette_score(self.X_scaled, labels))

        return K_range, wcss, silhouette_scores
    
    def run_kmeans(self, optimal_k):
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        self.df["Cluster"] = kmeans.fit_predict(self.X_scaled)
        
        score = silhouette_score(self.df[self.columns_normalize], self.df["Cluster"])
        
        self.summary = self.create_segment_profile()
        
        return self.df, kmeans, score

    def create_segment_profile(self):
        summary = self.df.groupby("Cluster")[self.summary_columns].mean().round(2)
        return summary

    def describe_segments(self):
        descriptions = {}
        for cluster in self.summary.index:
            row = self.summary.loc[cluster]
            desc = []
            if row["Income"] > self.summary["Income"].mean():
                desc.append("high income")
            else:
                desc.append("low income")

            if row["TotalSpend"] > self.summary["TotalSpend"].mean():
                desc.append("high spender")
            else:
                desc.append("low spender")

            if row["OnlinePurchaseRatio"] > 0.4:
                desc.append("digitally engaged")
                
            if row["ChurnRisk"] > 0.5:
                desc.append("churn risk")

            if row["Has_Kids"] > 0.5:
                desc.append("has children")
                
            if row["DiscountAffinity"] > self.summary["DiscountAffinity"].mean():
                desc.append("discount seeker")
                
            if row["CampaignEngaged"] > self.summary["CampaignEngaged"].mean():
                desc.append("campaign responsive")

            descriptions[cluster] = f"Segment {cluster}: {', '.join(desc).capitalize()}"
            
        return descriptions
    
    def generate_strategic_recommendations(self):
        recommendations = {}
        
        for cluster in self.summary.index:
            row = self.summary.loc[cluster]
            segment_recs = []
            
            # High-value customers
            if row["TotalSpend"] > self.summary["TotalSpend"].mean() and row["Income"] > self.summary["Income"].mean():
                segment_recs.append("Focus on loyalty programs and exclusive offerings")
                segment_recs.append("Consider premium product lines or services")
                
            # Online shoppers
            if row["OnlinePurchaseRatio"] > 0.4:
                segment_recs.append("Invest in personalized digital experiences")
                segment_recs.append("Optimize mobile app and website user journeys")
            else:
                segment_recs.append("Create incentives for online channel adoption")
                
            # Churn risk
            if row["ChurnRisk"] > 0.5:
                segment_recs.append("Develop targeted retention campaigns")
                segment_recs.append("Implement proactive customer service outreach")
                
            # Families
            if row["Has_Kids"] > 0.5:
                segment_recs.append("Create family-oriented promotions and bundles")
                segment_recs.append("Consider timing campaigns around school holidays")
                
            # Campaign responsive
            if row["CampaignEngaged"] > self.summary["CampaignEngaged"].mean():
                segment_recs.append("Increase campaign frequency to this segment")
                segment_recs.append("Test different campaign types to optimize engagement")
            else:
                segment_recs.append("Rethink campaign strategy for this segment")
                segment_recs.append("Consider alternative communication channels")
                
            # Discount seekers
            if row["DiscountAffinity"] > self.summary["DiscountAffinity"].mean():
                segment_recs.append("Create strategic discount promotions")
                segment_recs.append("Consider loyalty-based discount tiers")
            
            recommendations[cluster] = segment_recs
            
        return recommendations

# Helper function to create metrics
def display_metric(title, value, delta=None):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{title}</div>
    </div>
    """, unsafe_allow_html=True)

# Helper function for creating section containers with styling
def create_card(title, content_function):
    st.markdown(f"""
    <div class="dashboard-card">
        <h3>{title}</h3>
    </div>
    """, unsafe_allow_html=True)
    content_function()

# Main app
def main():
    # App header
    st.markdown("""
    <div class="main-header">
        <h1>Customer Segmentation Analysis Dashboard</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for file upload status
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
    
    # Handle file upload
    if not st.session_state.file_uploaded:
        uploaded_file = st.file_uploader("📤 Upload Customer Data (CSV)", type=["csv"], 
                                       help="Upload a CSV file containing customer data")
        
        if uploaded_file is not None:
            try:
                with st.spinner("Processing data..."):
                    df = pd.read_csv(uploaded_file)
                    st.session_state.df = df
                    st.session_state.file_uploaded = True
                    st.rerun()
            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.info("Please make sure your CSV file has the required columns for customer segmentation")
    
    # If file is uploaded, show minimized uploader and main dashboard
    if st.session_state.file_uploaded:
        # Show minimized file uploader with option to change file
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown('<div class="minimized-uploader">📄 CSV file uploaded successfully</div>', unsafe_allow_html=True)
        with col2:
            if st.button("Change File"):
                st.session_state.file_uploaded = False
                if 'df_clustered' in st.session_state:
                    del st.session_state['df_clustered']
                st.rerun()
        
        # Display raw data sample in an expander
        with st.expander("🔍 Preview Raw Data", expanded=False):
            st.dataframe(st.session_state.df.head(), use_container_width=True)
            st.text(f"Dataset Shape: {st.session_state.df.shape}")
        
        # Initialize segmentation
        segmentation = CustomerSegmentation(st.session_state.df)
        
        # Dashboard tabs - using horizontal layout
        tabs = st.tabs(["📊 Cluster Analysis", "👥 Segment Profiles", "📈 Customer Distribution", "💼 Strategic Recommendations"])
        
        # Tab 1: Cluster Analysis
        with tabs[0]:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### Find Optimal Number of Clusters")
                if st.button("Calculate Optimal K", key="calc_k"):
                    with st.spinner("Calculating optimal K..."):
                        k_range, wcss, silhouette_scores = segmentation.find_optimal_k()
                        
                        # Store in session state
                        st.session_state['k_range'] = k_range
                        st.session_state['wcss'] = wcss
                        st.session_state['silhouette_scores'] = silhouette_scores
                        
                        # Find best K based on silhouette score
                        best_k = k_range[silhouette_scores.index(max(silhouette_scores))]
                        st.session_state['best_k'] = best_k
            
            with col2:
                st.markdown("### Run K-means Clustering")
                
                # Check if we have calculated optimal K
                if 'best_k' in st.session_state:
                    default_k = st.session_state['best_k']
                else:
                    default_k = 3
                
                k_value = st.slider("Select number of clusters", min_value=2, max_value=10, value=default_k)
                
                if st.button("Run Clustering", key="run_cluster"):
                    with st.spinner("Running K-means clustering..."):
                        # Add a slight delay for better UX with the spinner
                        time.sleep(0.5)
                        df_clustered, kmeans, silhouette = segmentation.run_kmeans(k_value)
                        st.session_state['df_clustered'] = df_clustered
                        st.session_state['silhouette'] = silhouette
                        st.session_state['segment_summary'] = segmentation.summary
                        st.session_state['segment_descriptions'] = segmentation.describe_segments()
                        st.session_state['recommendations'] = segmentation.generate_strategic_recommendations()
            
            # Display results if they exist
            if 'k_range' in st.session_state:
                st.markdown("---")
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Plot elbow method
                    fig, ax1 = plt.subplots(figsize=(10, 6))
                    
                    color = '#4267B2'
                    ax1.set_xlabel('Number of Clusters (K)')
                    ax1.set_ylabel('WCSS (Within-Cluster Sum of Squares)', color=color)
                    ax1.plot(list(st.session_state['k_range']), st.session_state['wcss'], 'o-', color=color)
                    ax1.tick_params(axis='y', labelcolor=color)
                    
                    ax2 = ax1.twinx()
                    color = '#E14D2A'
                    ax2.set_ylabel('Silhouette Score', color=color)
                    ax2.plot(list(st.session_state['k_range']), st.session_state['silhouette_scores'], 'o-', color=color)
                    ax2.tick_params(axis='y', labelcolor=color)
                    
                    plt.title('Elbow Method and Silhouette Score')
                    fig.tight_layout()
                    st.pyplot(fig)
                
                with col2:
                    # Optimal K recommendation
                    st.markdown(f"""
                    <div class="success-msg">
                        <h3>Recommended Clusters: {st.session_state['best_k']}</h3>
                        <p>Based on highest silhouette score</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Silhouette score metrics
                    if 'silhouette' in st.session_state:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{st.session_state['silhouette']:.4f}</div>
                            <div class="metric-label">Silhouette Score</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Show cluster distribution if clustering is done
            if 'df_clustered' in st.session_state:
                st.markdown("---")
                st.markdown("### Customer Distribution Across Clusters")
                
                cluster_counts = st.session_state['df_clustered']['Cluster'].value_counts().sort_index()
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Display metrics about clusters
                    st.markdown("#### Cluster Statistics")
                    total_customers = len(st.session_state['df_clustered'])
                    num_clusters = len(cluster_counts)
                    
                    metrics_col1, metrics_col2 = st.columns(2)
                    with metrics_col1:
                        display_metric("Total Customers", f"{total_customers:,}")
                    with metrics_col2:
                        display_metric("Number of Clusters", num_clusters)
                    
                    # Display cluster sizes
                    st.markdown("#### Cluster Sizes")
                    for cluster, count in cluster_counts.items():
                        percentage = (count / total_customers) * 100
                        st.markdown(f"""
                        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                            <span>Cluster {cluster}</span>
                            <span><b>{count}</b> ({percentage:.1f}%)</span>
                        </div>
                        <div style="height: 8px; background-color: #e9ecef; border-radius: 4px;">
                            <div style="height: 100%; width: {percentage}%; background-color: {'#4267B2'}; border-radius: 4px;"></div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    # Distribution visualization
                    fig = px.bar(x=cluster_counts.index, y=cluster_counts.values, 
                                 labels={'x': 'Cluster', 'y': 'Number of Customers'},
                                 title="Customer Distribution",
                                 color=cluster_counts.index,
                                 color_continuous_scale=px.colors.sequential.Blues)
                    
                    fig.update_layout(
                        showlegend=False,
                        xaxis=dict(tickmode='linear'),
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Only show remaining tabs if clustering has been run
        if 'df_clustered' in st.session_state:
            # Tab 2: Segment Profiles
            with tabs[1]:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("### Customer Segment Profiles")
                    
                    # Format and display segment profiles with better styling
                    profile_df = st.session_state['segment_summary'].copy()
                    st.dataframe(profile_df, use_container_width=True, height=400)
                    
                    # Display segment descriptions
                    st.markdown("### Segment Descriptions")
                    for cluster, desc in st.session_state['segment_descriptions'].items():
                        st.markdown(f"""
                        <div style="background-color: #F44336; padding: 10px; border-radius: 5px; margin-bottom: 10px; border-left: 4px solid #4267B2;">
                            <b>{desc}</b>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### Segment Comparison")
                    
                    # Select features for radar chart
                    selected_features = st.multiselect(
                        "Select features to compare", 
                        segmentation.summary_columns,
                        default=["Income", "TotalSpend", "OnlinePurchaseRatio", "CampaignEngaged", "DiscountAffinity"]
                    )
                    
                    if selected_features:
                        # Normalize the data for radar chart
                        summary_normalized = st.session_state['segment_summary'][selected_features].copy()
                        for feature in selected_features:
                            max_val = summary_normalized[feature].max()
                            min_val = summary_normalized[feature].min()
                            if max_val > min_val:
                                summary_normalized[feature] = (summary_normalized[feature] - min_val) / (max_val - min_val)
                        
                        # Create radar chart
                        fig = go.Figure()
                        
                        colors = px.colors.qualitative.Bold
                        for i, cluster in enumerate(summary_normalized.index):
                            fig.add_trace(go.Scatterpolar(
                                r=summary_normalized.loc[cluster].values,
                                theta=selected_features,
                                fill='toself',
                                name=f'Cluster {cluster}',
                                line_color=colors[i % len(colors)]
                            ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1]
                                )
                            ),
                            showlegend=True,
                            title="Segment Comparison (Normalized Values)",
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Add a heatmap visualization for better comparison
                    if len(selected_features) > 0:
                        st.markdown("### Segment Heatmap")
                        
                        fig = px.imshow(
                            st.session_state['segment_summary'][selected_features],
                            labels=dict(x="Features", y="Cluster", color="Value"),
                            x=selected_features,
                            y=st.session_state['segment_summary'].index.astype(str),
                            color_continuous_scale="Blues",
                            aspect="auto"
                        )
                        
                        fig.update_layout(
                            xaxis=dict(tickangle=45),
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            
            # Tab 3: Customer Distribution
            with tabs[2]:
                st.markdown("### Customer Distribution Analysis")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Feature selection for scatter plot
                    x_feature = st.selectbox("Select X-axis feature", segmentation.summary_columns, index=0)
                    y_feature = st.selectbox("Select Y-axis feature", segmentation.summary_columns, index=2)
                    
                    # Key metrics distribution by cluster
                    st.markdown("### Key Metrics Distribution")
                    
                    selected_metric = st.selectbox(
                        "Select metric to analyze",
                        ["Income", "Age", "TotalSpend", "Tenure", "Recency", "SpendPerChild", "TotalPurchases", "ChurnRisk", "OnlinePurchaseRatio"]
                    )
                
                with col2:
                    # Create scatter plot with improved styling
                    fig = px.scatter(
                        st.session_state['df_clustered'], 
                        x=x_feature, 
                        y=y_feature, 
                        color='Cluster', 
                        opacity=0.7,
                        title=f"{y_feature} vs {x_feature} by Cluster",
                        color_continuous_scale=px.colors.qualitative.Bold
                    )
                    
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Box plot for distribution analysis
                fig = px.box(
                    st.session_state['df_clustered'],
                    x='Cluster',
                    y=selected_metric,
                    color='Cluster',
                    title=f"Distribution of {selected_metric} by Cluster",
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add a violin plot option
                show_violin = st.checkbox("Show Violin Plot")
                if show_violin:
                    fig = px.violin(
                        st.session_state['df_clustered'],
                        x='Cluster',
                        y=selected_metric,
                        color='Cluster',
                        box=True,
                        title=f"Violin Plot of {selected_metric} by Cluster",
                        color_discrete_sequence=px.colors.qualitative.Bold
                    )
                    
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Tab 4: Strategic Recommendations
            with tabs[3]:
                st.markdown("### Strategic Marketing Recommendations")
                
                # Select cluster for detailed recommendations
                selected_cluster = st.selectbox(
                    "Select customer segment for detailed recommendations",
                    list(st.session_state['recommendations'].keys()),
                    format_func=lambda x: f"Segment {x} - {st.session_state['segment_descriptions'][x].split(':')[1].strip()}"
                )
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Display segment profile
                    st.markdown(f"### Segment {selected_cluster} Profile")
                    st.markdown(f"**{st.session_state['segment_descriptions'][selected_cluster]}**")
                    
                    # Create metrics display for key characteristics
                    segment_data = st.session_state['segment_summary'].loc[selected_cluster]
                    
                    key_metrics_col1, key_metrics_col2 = st.columns(2)
                    with key_metrics_col1:
                        display_metric("Income", f"${segment_data['Income']:,.2f}")
                        display_metric("Total Spend", f"${segment_data['TotalSpend']:,.2f}")
                        display_metric("Online Ratio", f"{segment_data['OnlinePurchaseRatio']:.2%}")
                    
                    with key_metrics_col2:
                        display_metric("Churn Risk", f"{segment_data['ChurnRisk']:.2%}")
                        display_metric("Discount Affinity", f"{segment_data['DiscountAffinity']:.2%}")
                        display_metric("Campaign Engaged", f"{segment_data['CampaignEngaged']:.2%}")
                    
                    # Display recommendations with styled cards
                    st.markdown("### Recommended Strategies")
                    for rec in st.session_state['recommendations'][selected_cluster]:
                        st.markdown(f"""
                        <div style="background-color: #F44336; padding: 15px; border-radius: 8px; 
                            margin-bottom: 12px; border-left: 4px solid #4267B2; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <div style="font-weight: 500;">{rec}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    # Visual representation of segment characteristics
                    st.markdown("### Segment Characteristics")
                    
                    # Create horizontal bar chart for key metrics
                    metrics_to_show = ["Income", "TotalSpend", "OnlinePurchaseRatio", 
                                    "DiscountAffinity", "CampaignEngaged", "ChurnRisk"]
                    
                    # Calculate relative values compared to average
                    segment_values = []
                    for metric in metrics_to_show:
                        segment_val = segment_data[metric]
                        avg_val = st.session_state['segment_summary'][metric].mean()
                        rel_val = (segment_val / avg_val - 1) * 100  # percentage difference from average
                        segment_values.append({
                            'Metric': metric,
                            'Value': rel_val,
                            'AbsValue': abs(rel_val),
                            'Direction': 'Above average' if rel_val >= 0 else 'Below average'
                        })
                    
                    # Create dataframe for visualization
                    comparison_df = pd.DataFrame(segment_values)
                    
                    # Create horizontal bar chart
                    fig = px.bar(
                        comparison_df,
                        y='Metric',
                        x='Value',
                        color='Direction',
                        color_discrete_map={'Above average': '#4267B2', 'Below average': '#E14D2A'},
                        title="Comparison to Average (%)",
                        orientation='h',
                        text='AbsValue'
                    )
                    
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    
                    fig.update_layout(
                        xaxis_title="% Difference from Average",
                        yaxis_title="",
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Additional guidance section
                    st.markdown("### Implementation Guidance")
                    
                    implementation_guidance = {
                        "High value & low churn risk": "Focus on upselling premium products and loyalty rewards.",
                        "High value & high churn risk": "Implement proactive retention strategies and personalized outreach.",
                        "Low value & low churn risk": "Create graduated value enhancement offers to increase spending.",
                        "Low value & high churn risk": "Evaluate acquisition costs against potential lifetime value."
                    }
                    
                    # Determine which guidance to show based on segment characteristics
                    high_value = segment_data['TotalSpend'] > st.session_state['segment_summary']['TotalSpend'].mean()
                    high_churn = segment_data['ChurnRisk'] > st.session_state['segment_summary']['ChurnRisk'].mean()
                    
                    guidance_key = f"{'High' if high_value else 'Low'} value & {'high' if high_churn else 'low'} churn risk"
                    
                    st.markdown(f"""
                    <div style="background-color: #4267B2; padding: 20px; border-radius: 10px; margin-top: 20px;">
                        <h4>💡 {guidance_key}</h4>
                        <p>{implementation_guidance[guidance_key]}</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Add customer lookup tool at the bottom of tab 4
                st.markdown("---")
                st.markdown("### Customer Lookup Tool")
                st.markdown("Look up which segment a specific customer belongs to")

                # Get customer ID for lookup if available
                if 'ID' in st.session_state.df.columns:
                    customer_ids = st.session_state.df['ID'].tolist()
                    selected_id = st.selectbox("Select Customer ID", customer_ids)
                    
                    if st.button("Look Up Customer"):
                        with st.spinner("Looking up customer data..."):
                            customer_data = st.session_state.df[st.session_state.df['ID'] == selected_id]
                            
                            if not customer_data.empty:
                                customer_index = customer_data.index[0]
                                
                                if customer_index in st.session_state['df_clustered'].index:
                                    cluster = st.session_state['df_clustered'].loc[customer_index, 'Cluster']
                                    
                                    # Success message with segment info
                                    st.markdown(f"""
                                    <div class="success-msg">
                                        <h3>Customer {selected_id} belongs to {st.session_state['segment_descriptions'][cluster]}</h3>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Show customer key metrics
                                    customer_metrics = st.session_state['df_clustered'].loc[customer_index, segmentation.summary_columns]
                                    
                                    # Calculate how customer differs from segment average
                                    segment_avg = st.session_state['segment_summary'].loc[cluster]
                                    
                                    comparison_data = pd.DataFrame({
                                        'Customer': customer_metrics,
                                        'Segment Average': segment_avg,
                                        'Difference %': ((customer_metrics / segment_avg) - 1) * 100
                                    }).round(2)
                                    
                                    st.markdown("#### Customer Profile vs. Segment Average")
                                    st.dataframe(comparison_data, use_container_width=True)
                                    
                                    # Personalized recommendations
                                    st.markdown("#### Personalized Recommendations")
                                    st.markdown("Based on this customer's specific profile and segment:")
                                    
                                    # Example personalized recommendation logic
                                    recs = []
                                    
                                    if customer_metrics['TotalSpend'] > segment_avg['TotalSpend']:
                                        recs.append("High-value customer: Consider for VIP program and exclusive offers")
                                    
                                    if customer_metrics['ChurnRisk'] > segment_avg['ChurnRisk']:
                                        recs.append("Higher churn risk than segment average: Prioritize for retention campaign")
                                    
                                    if customer_metrics['OnlinePurchaseRatio'] < segment_avg['OnlinePurchaseRatio']:
                                        recs.append("Lower online engagement: Target with digital channel incentives")
                                        
                                    if customer_metrics['CampaignEngaged'] > segment_avg['CampaignEngaged']:
                                        recs.append("Responsive to campaigns: Include in next promotional wave")
                                    
                                    if not recs:
                                        recs.append("Customer profile aligns closely with segment average. Apply standard segment strategies.")
                                    
                                    for rec in recs:
                                        st.markdown(f"- {rec}")
                                else:
                                    st.error("Customer not found in clustered data")
                            else:
                                st.error("Customer ID not found")
                else:
                    st.info("Customer ID column not found in dataset")

                # Add option to download segmented data
                st.markdown("---")
                st.markdown("### Export Segmented Customer Data")

                if st.button("Prepare Download"):
                    with st.spinner("Preparing file for download..."):
                        # Create Excel file in memory
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            # Export segmented customer data
                            st.session_state['df_clustered'].to_excel(writer, sheet_name='Segmented Customers', index=False)
                            
                            # Export segment summary
                            st.session_state['segment_summary'].to_excel(writer, sheet_name='Segment Profiles')
                            
                            # Create recommendations sheet
                            rec_rows = []
                            for cluster, recs in st.session_state['recommendations'].items():
                                for rec in recs:
                                    rec_rows.append({
                                        'Segment': cluster,
                                        'Description': st.session_state['segment_descriptions'][cluster],
                                        'Recommendation': rec
                                    })
                            
                            pd.DataFrame(rec_rows).to_excel(writer, sheet_name='Recommendations', index=False)
                            
                        # Download link
                        st.markdown("#### Download Segmented Customer Data")
                        st.download_button(
                            label="Download Excel File",
                            data=output.getvalue(),
                            file_name="customer_segmentation_results.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

# Add this line to properly close the main function
if __name__ == "__main__":
    main()