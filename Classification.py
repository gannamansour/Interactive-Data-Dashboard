import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_option_menu import option_menu  
from contextlib import contextmanager, redirect_stdout
from io import StringIO
# Clustering :
from sklearn.cluster import KMeans

# Label Encoding :
from sklearn.preprocessing import LabelEncoder

# Model Training :
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# For Balancing 
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

# K NN Neighbours
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def app():
    st.title("Classification For Telescope Data")

    # Output in Web Streamlit :
@contextmanager
def st_capture(output_func):
                with StringIO() as stdout, redirect_stdout(stdout):
                    old_write = stdout.write

                    def new_write(string):
                        ret = old_write(string)
                        output_func(stdout.getvalue())
                        return ret
                    
                    stdout.write = new_write
                    yield

data = pd.read_csv(r"D:\Knn-LinearRegression-main\Pages\Telescope_data.csv")
data = pd.DataFrame(data)

with st.sidebar:
    selected = option_menu(
        menu_title="User's Input For Data Preparation",
        options=[
            "Show Data",
            "Balancing Telescope Data",
            "Spliting Data",
            "K-NN Classification",
            "K-Mean Clustering",
            "model accuracy, precision, recall and f-score"
        ],
    )

if selected == "Show Data":
        st.title("Dashboard For Telescope Data")
        st.write("You selected All Data")
        st.dataframe(data)

if selected == "Balancing Telescope Data":
        st.title("Balancing Dataset")
        x = data.drop(['class'], axis=1)
        y = data['class']

        st.title("Class distribution")
        output = st.empty()
        with st_capture(output.code):
            print(y.value_counts())

        fig1, ax1 = plt.subplots()
        y.value_counts().plot.pie(autopct='%.2f', ax=ax1)
        ax1.set_title("Class distribution")
        st.pyplot(fig1)

        st.title("Random Undersampling")
        rus = RandomUnderSampler(sampling_strategy=1)
        X_res, y_res = rus.fit_resample(x, y)
        fig2, ax2 = plt.subplots()
        y_res.value_counts().plot.pie(autopct='%.2f', ax=ax2)
        ax2.set_title("Under-sampling")
        st.pyplot(fig2)
        output = st.empty()
        with st_capture(output.code):
            print(y_res.value_counts())

        st.title("Random Oversampling")
        ros = RandomOverSampler(sampling_strategy="not majority")
        X_res, y_res = ros.fit_resample(x, y)
        fig3, ax3 = plt.subplots()
        y_res.value_counts().plot.pie(autopct='%.2f', ax=ax3)
        ax3.set_title("Over-sampling")
        st.pyplot(fig3)
        output = st.empty()
        with st_capture(output.code):
            print(y_res.value_counts())

if selected == "K-Mean Clustering":

    with st.sidebar:
        selected_Clus_Data = option_menu(
            menu_title="User's Input For a specific data clustering with Wave length",
            options=[
                "fWidth",
                "fSize",
                "fConc",
                "fConc1",
                "fAsym",
                "fM3Long",
                "fM3Trans",
                "fAlpha",
                "fDist"
            ],
        )

    st.title("K-Mean Clustering for Wave Length and User's input")
    st.write("Select Number Of Clusters:")

    Number_Of_Clusters = st.slider(
        label='Number Of Clusters',
        min_value=1,
        max_value=50,
        value=0,
        step=1
    )

    Button = st.button("Show K Mean:")

    if Button:
        # Apply K-Means Clustering
        kmeans = KMeans(n_clusters=Number_Of_Clusters, random_state=42)
        selected_feature = selected_Clus_Data  # Directly use the selected option
        cluster = kmeans.fit_predict(data[['fLength', selected_feature]])

        st.title("Clusters:")
        st.write(cluster)

        centroids = kmeans.cluster_centers_
        st.title("Centroids:")
        st.write(centroids)

        st.title("Visualization For Clusters:")
        features = data[['fLength', selected_feature]].values

        # Scatter plot with proper x and y values
        fig, ax = plt.subplots()
        scatter = ax.scatter(features[:, 0], features[:, 1], c=cluster, cmap='viridis', s=30)
        ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100, label='Centroids')
        ax.set_title("K-Means Clustering")
        ax.set_xlabel("fLength")
        ax.set_ylabel(selected_feature)
        ax.legend()
        st.pyplot(fig)

if selected == "model accuracy, precision, recall and f-score":
    
    # Label encoding
    for col in data.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    # Streamlit UI
    st.title("Training and Testing Models")
    st.write("Refers to the process of assessing a machine learning model's performance to ensure it meets efficiency standards.")
    st.write("This evaluation is critical to determining whether the model is suitable for deployment or requires improvement.")

    # User inputs
    Randnom_State_User = st.slider(
        label='Random State for Train-Testing Splitting',
        min_value=0,
        max_value=50,
        value=42,
        step=1
    )

    Test_Size_User = st.slider(
        label='Test Size for Train-Testing Splitting',
        min_value=0.1,
        max_value=0.9,
        value=0.2,
        step=0.1
    )

    # Sidebar option menu
    with st.sidebar:
        selected = option_menu(
            menu_title="Target Column Selection",
            options=[
                "fWidth",
                "fSize",
                "fConc",
                "fConc1",
                "fAsym",
                "fM3Long",
                "fM3Trans",
                "fAlpha",
                "fDist"
            ],
        )

    # Display selected target column
    st.write(f"Selected Option for Testing Model: {selected}")

    button = st.button("Test Training Model")

    if button:
        # Split data
        x_train = data.drop(columns=[selected])
        y_train = data[selected]

        # Train-test split
        x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(
            x_train, y_train, test_size=Test_Size_User, random_state=Randnom_State_User
        )

        # Data shape
        st.title("Splitting Data Shape")
        output = st.empty()
        with st_capture(output.code):
            print("Training Data Shape:", x_train_split.shape)
            print("Validation Data Shape:", x_val_split.shape)

        # Decision Tree Classifier
        clf = DecisionTreeClassifier(random_state=Randnom_State_User)
        clf.fit(x_train_split, y_train_split)

        # Predictions and metrics
        y_pred = clf.predict(x_val_split)
        accuracy = accuracy_score(y_val_split, y_pred)
        precision = precision_score(y_val_split, y_pred, average='weighted')
        recall = recall_score(y_val_split, y_pred, average='weighted')
        f1 = f1_score(y_val_split, y_pred, average='weighted')

        # Display metrics
        st.title("Model Evaluation Metrics")
        output = st.empty()
        with st_capture(output.code):
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")

        # K-Fold Cross-Validation
        st.title("K-Fold Cross-Validation")
        kf = KFold(n_splits=5, shuffle=True, random_state=Randnom_State_User)

        accuracies, precisions, recalls, f1_scores = [], [], [], []

        for train_index, val_index in kf.split(x_train):
            x_train_fold, x_val_fold = x_train.iloc[train_index], x_train.iloc[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

            clf.fit(x_train_fold, y_train_fold)
            y_pred = clf.predict(x_val_fold)

            accuracies.append(accuracy_score(y_val_fold, y_pred))
            precisions.append(precision_score(y_val_fold, y_pred, average='weighted'))
            recalls.append(recall_score(y_val_fold, y_pred, average='weighted'))
            f1_scores.append(f1_score(y_val_fold, y_pred, average='weighted'))

        # Display cross-validation metrics
        st.title("Average Metrics Across Folds")
        output = st.empty()
        with st_capture(output.code):
            print(f"Average Accuracy: {np.mean(accuracies):.4f}")
            print(f"Average Precision: {np.mean(precisions):.4f}")
            print(f"Average Recall: {np.mean(recalls):.4f}")
            print(f"Average F1 Score: {np.mean(f1_scores):.4f}")
    
if selected == "K-NN Classification":
    
    # User selects the number of neighbors
    N_Neighbors = st.slider(
        label='Number of Neighbors (K)',
        min_value=1,  # Min should be 1 (0 is invalid for KNN)
        max_value=50,
        value=5,
        step=1
    )
    
    button = st.button("Test Training Model")

    if button:
        # Drop the specified column (Ensure 'flength' is correct)
        if 'flength' in data.columns:
            data = data.drop(columns=['flength'])

        # Splitting features and target
        x = data.iloc[:, :-1]  # Automatically select all feature columns except last
        y = data.iloc[:, -1]   # Last column as the target variable

        # Train-test split
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=11, test_size=0.2)

        # Feature scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)  # Correct usage of transform

        # KNN Classifier with user-selected neighbors
        knn = KNeighborsClassifier(n_neighbors=N_Neighbors)
        knn.fit(x_train, y_train)

        # Predictions
        st.title("KNN Predictions")
        with st.expander("Predicted Labels"):
            y_pred = knn.predict(x_test)
            st.code(y_pred)

        # Model Accuracy
        st.title("Model Accuracy Score")
        with st.expander("Accuracy Score"):
            score = knn.score(x_test, y_test)
            st.code(score)

        # Confusion Matrix
        st.title("Confusion Matrix")
        with st.expander("Confusion Matrix Output"):
            cm = confusion_matrix(y_test, y_pred)
            st.code(cm)

        # Classification Report
        st.title("Classification Report")
        with st.expander("Classification Report Output"):
            cr = classification_report(y_test, y_pred)
            st.code(cr)

        # Number of Samples Used in Training
        st.title("Training Sample Count")
        with st.expander("Number of Samples Used for Training"):
            st.code(knn.n_samples_fit_)
