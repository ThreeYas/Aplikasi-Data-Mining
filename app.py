import streamlit as st
# mengimpor paket pandas kemudian diberi nama alias pd
import pandas as pd
# mengimpor paket numpy kemudian diberi nama alias np
import numpy as np
# min max scaler untuk normalisasi
from sklearn.preprocessing import MinMaxScaler
# library untuk Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer, accuracy_score,precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score,recall_score,f1_score
#Model Select
from sklearn.model_selection import train_test_split

#Mengonfigurasi pengaturan default halaman.
st.set_page_config(page_title="Triyas Septiyanto", page_icon='icon.png')

st.title("APLIKASI DATA MINING (Studi Kasus Peminjaman Uang)")
st.write("Triyas Septiyanto | 200411100043 | Penambangan Data A")

ket_data , dataset, preprocessing, modeling, implementasi = st.tabs(["Keterangan Data","Dataset", "Preprocessing", "Modeling", "Implementasi"])

with ket_data:
    st.write("""
            * Kode_Kontrak :
            Kolom kode_kontrak merupakan identitas dari sebuah data. identitas data tidak perlu diikutkan untuk klasifikasi. Sehingga kolom ini akan diabaikan saat klasifikasi data.Pada kolom ini merupakan kolom identitas pada nasabah yang menjadi pelanggan.
            * Pendapatan Setahun (Juta):
            Kolom pendapatan_setahun_juta merupakan data yang bertype numerik. Ciri-ciri data bertype numerik adalah data tersebut bernilai angka dan bertype integer. Pada kolom ini merupakan kolom pendapatan nasabah pada satu tahun dan memiliki satuan juta.
            * KPR Aktif :
            [YA, TIDAK]
            Kolom kpr_aktif merupakan data bertype categorial/oridinal sehingga harus di normalisasikan agar berubah menjadi numerik. Ciri-ciri data bertype categorial yaitu data tersebut merupakan nama dari suatu hal, dan ciri data bertype ordinal yaitu memiliki 2 keadaan. Pada kolom ini merupakan kolom kredit pemilik rumah yang dimiliki nasabah. jika nasabah tersebut mempunyai KPR aktif, maka data tersebut bernilai "Ya"
            * Durasi Pinjaman (Bulan) :
            [12 Bulan, 24 Bulan, 36 Bulan, 48 Bulan]
            Kolom durasi_pinjaman_bulan data bertype numerik. Ciri-ciri data bertype numerik adalah data tersebut bernilai angka dan bertype integer. Pada kolom ini merupakan kolom durasi pinjaman yang akan diajukan oleh nasabah dalam satuan bulan.
            * Jumlah Tanggungan :
            [0 - 6 Tanggungan]
            Kolom jumlah_tanggungan merupakan data bertype numerik. Ciri-ciri data bertype numerik adalah data tersebut bernilai angka dan bertype integer. Pada kolom ini merupakan jumlah tanggungan yang dimiliki nasaban saat dirumah.
            * Rentan Hari (rata_rata_overdue): 
            [0 - 30 hari, 31 - 45 hari, 46 - 60 hari, 61 - 90 hari, > 90 hari]
            Kolom rata_rata_overdue merupakan data bertype categorial sehingga harus di normalisasikan agar berubah menjadi numerik. Ciri-ciri data bertype categorial yaitu data tersebut merupakan nama dari suatu hal, Pada kolom ini merupakan kolom jangka pengembalian yang dipinjam oleh nasabah dalam waktu rentang yang sudah disediakan.
            * Nilai Resiko (risk_rating):
            [1 - 5]
            Kolom risk_rating merupakan class dengan type data numerik. Ciri-ciri data bertype numerik adalah data tersebut bernilai angka dan bertype integer. Kolom ini merupakan kelas dari data. dan yang akan di prediksi nanti merupakan data yang ada pada kolom ini.
            """)

with dataset:
    st.write("Dataset Credit Score Dari Kaggle berbentuk XLSX")
    kaggle = "https://www.kaggle.com/code/kerneler/starter-credit-scoring-c358cfde-2/data"
    st.markdown(f'[Dataset Credit Score Kaggle ]({kaggle})')
    st.write("Dataset Credit Score Dari Github berbentuk CSV")
    github = "https://raw.githubusercontent.com/ThreeYas/credit-scoring/main/credit-scoring.csv"
    st.markdown(f'[Dataset Credit Score Github ]({github})')

    # membaca dataset csv dari url
    url = 'https://raw.githubusercontent.com/ThreeYas/credit-scoring/main/credit-scoring.csv'
    data = pd.read_csv(url)
    st.dataframe(data)

with preprocessing:
    st.write(" # Preprocessing ")
    data.head()
    # menghapus kolom risk_rating dari tabel
    X = data.drop(columns=["risk_rating"])

    # periksa apakah variabel target telah dihapus
    X.head()

    # memecah setiap kelas pada kolom "rata_rata_overdue" menjadi kolom tersendiri
    split_kolom_overdue = pd.get_dummies(X["rata_rata_overdue"], prefix="overdue")
    X = X.join(split_kolom_overdue)

    # menghapus kolom rata_rata_overdue dari tabel
    X = X.drop(columns = "rata_rata_overdue")

    # memilih fitur kelas
    label = data["risk_rating"]

    # memecah setiap kelas pada kolom "kpr_aktif" menjadi kolom tersendiri
    split_kolom_kpr = pd.get_dummies(X["kpr_aktif"], prefix="KPR")
    X = X.join(split_kolom_kpr)

    # menghapus kolom kpr_aktif dari tabel
    X = X.drop(columns = "kpr_aktif")

    st.write("Data preprocessing adalah proses yang mengubah data mentah ke dalam bentuk yang lebih mudah dipahami. Proses ini penting dilakukan karena data mentah sering kali tidak memiliki format yang teratur.")

    st.write("menampilkan dataframe dimana rata-rata overdue, risk rating dan kpr aktif sudah di hapus dari data")
    st.dataframe(X)

    st.write(" ## Normalisasi ")
    st.write('Normalisasi Kolom ', 'pendapatan_setahun_juta', 'durasi_pinjaman_bulan', 'jumlah_tanggungan')
    label_lama = ['pendapatan_setahun_juta', 'durasi_pinjaman_bulan', 'jumlah_tanggungan']
    label_baru = ['new_pendapatan_setahun_juta', 'new_durasi_pinjaman_bulan', 'new_jumlah_tanggungan']
    normalisasi_kolom = data[label_lama]

    st.dataframe(normalisasi_kolom)

    # Inisialisasi Fungsi MinMaxScaler
    scaler = MinMaxScaler()

    # Hitung minimum dan maksimum yang akan digunakan untuk penskalaan nanti.
    scaler.fit(normalisasi_kolom)

    # Skala fitur X menurut feature_range.
    kolom_ternormalisasi = scaler.transform(normalisasi_kolom)

    # Membuat DataFrame dari array
    df_kolom_ternormalisasi = pd.DataFrame(kolom_ternormalisasi, columns = label_baru)

    st.write("data setelah dinormalisasi")
    st.dataframe(df_kolom_ternormalisasi)

    # # menghapus kolom label_lama dari tabel
    X = X.drop(columns = label_lama)

    X = X.join(df_kolom_ternormalisasi)

    X = X.join(label)

    st.write("dataframe X baru")
    st.dataframe(X)

    label_subjek = ["Unnamed: 0",  "kode_kontrak"]
    X = X.drop(columns = label_subjek)

    st.write("dataframe X baru yang tidak ada fitur/kolom unnamed: 0 dan kode kontrak")
    st.dataframe(X)
    st.write("## Hitung Data")
    st.write("- Pisahkan kolom risk rating dari data frame")
    st.write("- Ambil kolom 'risk rating' sebagai target kolom untuk kategori kelas")
    st.write("- Pisahkan data latih dengan data tes")
    st.write("""            Spliting Data
                data latih (nilai data)
                X_train 
                data tes (nilai data)
                X_test 
                data latih (kelas data)
                y_train
                data tes (kelas data)
                y_test""")


    # memisahkan data risk_rating
    X = X.iloc[:, :-1]
    y = data.loc[:, "risk_rating"]
    y = data["risk_rating"].values

    # membagi data menjadi set train dan test (70:30)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)

    st.write("Menampilkan X")
    st.write(X)
    
    st.write("Menampilkan Y")
    st.write(y)
