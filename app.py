import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Dictionary untuk menampilkan keterangan dan terjemahan
descriptions = {
    'cap-shape': {'b': 'bell (lonceng)', 'c': 'conical (kerucut)', 'x': 'convex (cembung)', 'f': 'flat (datar)', 'k': 'knobbed (berbenjol)', 's': 'sunken (cekung)'},
    'cap-surface': {'f': 'fibrous (berserat)', 'g': 'grooves (beralur)', 'y': 'scaly (bersisik)', 's': 'smooth (halus)'},
    'cap-color': {'n': 'brown (coklat)', 'b': 'buff (kekuningan)', 'c': 'cinnamon (kayu manis)', 'g': 'gray (abu-abu)', 'r': 'green (hijau)', 'p': 'pink (merah muda)', 'u': 'purple (ungu)', 'e': 'red (merah)', 'w': 'white (putih)', 'y': 'yellow (kuning)'},
    'bruises': {'t': 'bruises (memar)', 'f': 'no (tidak)'},
    'odor': {'a': 'almond (almond)', 'l': 'anise (adas manis)', 'c': 'creosote (kreosot)', 'y': 'fishy (amis)', 'f': 'foul (busuk)', 'm': 'musty (apek)', 'n': 'none (tidak ada)', 'p': 'pungent (menyengat)', 's': 'spicy (pedas)'},
    'gill-attachment': {'a': 'attached (terikat)', 'd': 'descending (turun)', 'f': 'free (bebas)', 'n': 'notched (berlekuk)'},
    'gill-spacing': {'c': 'close (rapat)', 'w': 'crowded (penuh sesak)', 'd': 'distant (jauh)'},
    'gill-size': {'b': 'broad (lebar)', 'n': 'narrow (sempit)'},
    'gill-color': {'k': 'black (hitam)', 'n': 'brown (coklat)', 'b': 'buff (kekuningan)', 'h': 'chocolate (coklat)', 'g': 'gray (abu-abu)', 'r': 'green (hijau)', 'o': 'orange (oranye)', 'p': 'pink (merah muda)', 'u': 'purple (ungu)', 'e': 'red (merah)', 'w': 'white (putih)', 'y': 'yellow (kuning)'},
    'stalk-shape': {'e': 'enlarging (melebar)', 't': 'tapering (meruncing)'},
    'stalk-root': {'b': 'bulbous (bulat)', 'c': 'club (tongkat)', 'u': 'cup (cangkir)', 'e': 'equal (sama)', 'z': 'rhizomorphs (rizomorfa)', 'r': 'rooted (berakar)'},
    'stalk-surface-above-ring': {'f': 'fibrous (berserat)', 'y': 'scaly (bersisik)', 'k': 'silky (sutra)', 's': 'smooth (halus)'},
    'stalk-surface-below-ring': {'f': 'fibrous (berserat)', 'y': 'scaly (bersisik)', 'k': 'silky (sutra)', 's': 'smooth (halus)'},
    'stalk-color-above-ring': {'n': 'brown (coklat)', 'b': 'buff (kekuningan)', 'c': 'cinnamon (kayu manis)', 'g': 'gray (abu-abu)', 'o': 'orange (oranye)', 'p': 'pink (merah muda)', 'e': 'red (merah)', 'w': 'white (putih)', 'y': 'yellow (kuning)'},
    'stalk-color-below-ring': {'n': 'brown (coklat)', 'b': 'buff (kekuningan)', 'c': 'cinnamon (kayu manis)', 'g': 'gray (abu-abu)', 'o': 'orange (oranye)', 'p': 'pink (merah muda)', 'e': 'red (merah)', 'w': 'white (putih)', 'y': 'yellow (kuning)'},
    'veil-color': {'n': 'brown (coklat)', 'o': 'orange (oranye)', 'w': 'white (putih)', 'y': 'yellow (kuning)'},
    'ring-number': {'n': 'none (tidak ada)', 'o': 'one (satu)', 't': 'two (dua)'},
    'ring-type': {'e': 'evanescent (cepat hilang)', 'f': 'flaring (melebar)', 'l': 'large (besar)', 'n': 'none (tidak ada)', 'p': 'pendant (tergantung)', 's': 'sheathing (membungkus)', 'z': 'zone (zona)'},
    'spore-print-color': {'k': 'black (hitam)', 'n': 'brown (coklat)', 'b': 'buff (kekuningan)', 'h': 'chocolate (coklat)', 'r': 'green (hijau)', 'o': 'orange (oranye)', 'u': 'purple (ungu)', 'w': 'white (putih)', 'y': 'yellow (kuning)'},
    'population': {'a': 'abundant (melimpah)', 'c': 'clustered (berkelompok)', 'n': 'numerous (banyak)', 's': 'scattered (tersebar)', 'v': 'several (beberapa)', 'y': 'solitary (sendiri)'},
    'habitat': {'g': 'grasses (rumput)', 'l': 'leaves (daun)', 'm': 'meadows (padang rumput)', 'p': 'paths (jalur)', 'u': 'urban (kota)', 'w': 'waste (limbah)', 'd': 'woods (hutan)'}
}

feature_descriptions = {
    'cap-shape': 'Bentuk Tutup',
    'cap-surface': 'Permukaan Tutup',
    'cap-color': 'Warna Tutup',
    'bruises': 'Memar',
    'odor': 'Bau',
    'gill-attachment': 'Lampiran Insang',
    'gill-spacing': 'Jarak Insang',
    'gill-size': 'Ukuran Insang',
    'gill-color': 'Warna Insang',
    'stalk-shape': 'Bentuk Batang',
    'stalk-root': 'Akar Batang',
    'stalk-surface-above-ring': 'Permukaan Batang Di Atas Cincin',
    'stalk-surface-below-ring': 'Permukaan Batang Di Bawah Cincin',
    'stalk-color-above-ring': 'Warna Batang Di Atas Cincin',
    'stalk-color-below-ring': 'Warna Batang Di Bawah Cincin',
    'veil-color': 'Warna Kerudung',
    'ring-number': 'Jumlah Cincin',
    'ring-type': 'Tipe Cincin',
    'spore-print-color': 'Warna Cetakan Spora',
    'population': 'Populasi',
    'habitat': 'Habitat'
}

# Function to load and preprocess data
@st.cache
def load_data():
    url = "mushrooms.csv"
    data = pd.read_csv(url)
    data = data.drop('veil-type', axis=1)
    modus_stalk_root = data[data['stalk-root'] != '?']['stalk-root'].mode()[0]
    data['stalk-root'] = data['stalk-root'].replace('?', modus_stalk_root)
    for column in data.columns:
        data[column] = data[column].astype('category').cat.codes
    return data, modus_stalk_root

# Load data
data, modus_stalk_root = load_data()

# Split data
X = data.drop('class', axis=1)
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Link dataset
url_dataset = "https://www.kaggle.com/datasets/uciml/mushroom-classification"

# Streamlit app
st.title("Klasifikasi Jamur")
st.write(f"Akurasi Model: {accuracy:.2f}")
st.markdown(f"Link Dataset: [{url_dataset}]({url_dataset})")

# User input for prediction
st.header("Prediksi Keberacunan Jamur")

def user_input_features():
    input_data = {}
    original_data = pd.read_csv("mushrooms.csv").drop('veil-type', axis=1)
    original_data['stalk-root'] = original_data['stalk-root'].replace('?', modus_stalk_root)
    mappings = {column: {cat: i for i, cat in enumerate(original_data[column].astype('category').cat.categories)} for column in original_data.columns}

    for feature in X.columns:
        options = list(descriptions[feature].keys())
        option = st.selectbox(f"Pilih {feature_descriptions[feature]} ({feature}):", options, format_func=lambda x: f"{x}: {descriptions[feature][x]}")
        if option in mappings[feature]:
            input_data[feature] = mappings[feature][option]
        else:
            st.error(f"Nilai {option} tidak ditemukan dalam pemetaan untuk {feature}")
    return pd.DataFrame([input_data])

user_input = user_input_features()
if st.button("Prediksi"):
    prediction = rf.predict(user_input)
    result = 'Beracun' if prediction[0] == 1 else 'Dapat Dimakan'
    st.write(f"Jamur tersebut: {result}")
