import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import seaborn as sns

# Load the dataset
def load_data():
    file_path = "Clustering.csv"  # Replace with the correct path
    return pd.read_csv(file_path)

data = load_data()

# Streamlit app
def main():
    st.set_page_config(page_title="Hierarchical Clustering", layout="wide")
    st.title("🌐 Hierarchical Clustering Web App")

    # Sidebar for navigation
    st.sidebar.header("Settings")
    st.sidebar.markdown("Use the options below to configure clustering.")

    # Display dataset
    st.subheader("📊 Dataset Preview")
    st.write("Preview of the dataset:")
    st.dataframe(data.head(), use_container_width=True)

    # Select features for clustering
    st.subheader("⚙️ Select Features")
    columns = data.columns.tolist()
    columns.remove("ID")  # Exclude ID column by default

    selected_features = st.multiselect("Choose features for clustering:", columns, default=columns)

    if not selected_features:
        st.warning("Please select at least one feature for clustering.")
        return

    # Preprocess data
    clustering_data = data[selected_features]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_data)

    # Perform hierarchical clustering
    st.subheader("📈 Dendrogram")
    linkage_method = st.sidebar.selectbox(
        "Select linkage method:", ["ward", "complete", "average", "single"], index=0
    )

    st.write("Hierarchical clustering dendrogram using the selected features:")
    plt.figure(figsize=(12, 8))
    plt.title("Dendrogram")
    dendrogram = sch.dendrogram(sch.linkage(scaled_data, method=linkage_method))
    st.pyplot(plt)

    # Specify the number of clusters
    st.subheader("🌀 Clustering Results")
    num_clusters = st.slider("Select number of clusters:", min_value=2, max_value=10, value=3)

    # Fit model and assign clusters
    from scipy.cluster.hierarchy import fcluster

    clusters = fcluster(sch.linkage(scaled_data, method=linkage_method), num_clusters, criterion="maxclust")
    data["Cluster"] = clusters

    st.write("Dataset with cluster labels:")
    st.dataframe(data, use_container_width=True)

    # Visualize clusters
    st.subheader("🎨 Cluster Visualization")
    if len(selected_features) >= 2:
        x_axis = st.selectbox("Select X-axis feature:", selected_features, index=0)
        y_axis = st.selectbox("Select Y-axis feature:", selected_features, index=1)

        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x=data[x_axis], y=data[y_axis], hue=data["Cluster"], palette="tab10", s=100, alpha=0.8
        )
        plt.title("Clusters Visualization")
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.legend(title="Cluster")
        st.pyplot(plt)
    else:
        st.warning("Please select at least two features to visualize clusters.")

if __name__ == "__main__":
    main()
