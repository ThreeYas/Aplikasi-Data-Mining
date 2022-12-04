import streamlit as st
# mengimpor paket pandas kemudian diberi nama alias pd
import pandas as pd
# mengimpor paket numpy kemudian diberi nama alias np
import numpy as np
import joblib
# min max scaler untuk normalisasi
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
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
    
    st.write("## Penjelasan Dataset")
    st.write("""
            Data ini digunakan untuk Studi Kasus Peminjaman Uang

            Credit risk adalah resiko yang harus ditanggung oleh seorang individu atau lembaga ketika memberikan pinjaman - biasanya dalam bentuk uang - ke individu atau pihak lain. Resiko ini berupa tidak bisa dibayarkannya pokok dan bunga pinjaman, sehingga mengakibatkan kerugian berikut: — gangguan aliran kas (cash flow) sehingga modal kerja terganggu. — meningkatkan biaya operasional untuk mengejar pembayaran tersebut (collection). Untuk memperkecil resiko kredit ini, biasanya dilakukan proses yang disebut dengan credit scoring dan credit rating terhadap pihak peminjam. Output proses ini akan menjadi basis untuk menentukan apakah aplikasi pengajuan pinjaman baru diterima atau ditolak.

            Credit score adalah nilai resiko yang diberikan kepada seorang individu atau organisasi yang mengajukan pinjaman berdasarkan rekam jejak pinjaman dan pembayaran yang dilakukan. Proses pemberian credit score ini biasanya disebut sebagai credit scoring. Perhitungan credit score biasanya dibuat berdasarkan data historis lamanya keterlambatan pembayaran dan yang tidak bayar sama sekali (bad debt). Bad debt biasanya mengakibatkan lembaga pemberian kredit harus menyita aset atau melakukan write off. Nilai credit score biasanya bervariasi antar lembaga. Namun banyak yang kemudian mengadopsi model FICO Score yang memiliki rentang nilai 300 - 850. Semakin tinggi nilai yang didapatkan, maka semakin baik tingkat kemampuan seseorang atau sebuah lembaga untuk membayar pinjaman.

            Kadang banyak lembaga yang menggunakan risk rating atau tingkat resiko. Terbalik dengan credit score, semakin tinggi rating ini menunjukkan resiko yang semakin meningkat. Selain itu kodifikasi juga dibuat lebih simpel dibandingkan rentang nilai sehingga keputusan yang bisa diambil lebih cepat. Contoh, misalkan penggunaan kombinasi seperti huruf AAA, AA+, P-1, dan seterusnya. Atau untuk banyak internal lembaga peminjam, kategorisasi hanya menggunakan rentang angka yang kecil misalkan 1 sampai dengan 5.
            """)

    st.write("## Sumber Dataset")
    st.write("Dataset Credit Score Dari Kaggle berbentuk XLSX")
    kaggle = "https://www.kaggle.com/code/kerneler/starter-credit-scoring-c358cfde-2/data"
    st.markdown(f'[Link Dataset Credit Score Kaggle ]({kaggle})')
    st.write("Dataset Credit Score Dari Github berbentuk CSV")
    github = "https://raw.githubusercontent.com/ThreeYas/credit-scoring/main/credit-scoring.csv"
    st.markdown(f'[Link Dataset Credit Score Github ]({github})')

    st.write("## Penjelasan Kolom")
    st.write("""
            * Kode Kontrak :
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
    
    st.write("## Repository Github")
    st.write(" Kode ini dapat di akses di Github ThreeYas")
    repo = "https://github.com/ThreeYas/Aplikasi-Data-Mining"
    st.markdown(f'[ Link Repository Github ]({repo})')

with dataset:
    st.write("Dataset ini terdiri dari 900 baris dan 8 kolom")
    # membaca dataset csv dari url
    url = 'https://raw.githubusercontent.com/ThreeYas/credit-scoring/main/credit-scoring.csv'
    data = pd.read_csv(url)
    st.dataframe(data)

with preprocessing:
    st.write(" # Preprocessing ")
    st.write("Data yang belum dinormalisasi")
    st.dataframe(data)
    
    # memilih fitur kelas
    label = data["risk_rating"]

    st.write("Perintah DROP COLUMN digunakan untuk menghapus kolom di tabel yang sudah ada.")
    st.write("Menghapus kolom risk_rating dari tabel")
    # menghapus kolom risk_rating dari tabel
    X = data.drop(columns=["risk_rating"])

    # periksa apakah variabel target telah dihapus
    st.dataframe(X)

    st.write("""
            Mengubah kolom rata_rata_overdue menjadi tipe data numerik
            
            Memisahkan kolom numerik berdasarkan rentan hari

            
            # Rentan Hari
            [0 - 30 hari, 31 - 45 hari, 46 - 60 hari, 61 - 90 hari, > 90 hari]
            
            """)

    st.write("Join adalah fungsi bawaan yang digunakan untuk menggabungkan atau menggabungkan DataFrame yang berbeda")

    # memecah setiap kelas pada kolom "rata_rata_overdue" menjadi kolom tersendiri
    split_kolom_overdue = pd.get_dummies(X["rata_rata_overdue"], prefix="overdue")
    X = X.join(split_kolom_overdue)

    # menghapus kolom rata_rata_overdue dari tabel
    X = X.drop(columns = "rata_rata_overdue")

    st.write("""
            Kemudian normalisasi kolom 

            Memisahkan kolom numerik berdasarkan "YA" atau "TIDAK"

            
            # Value kpr_aktif
            # ['YA', 'TIDAK']
            
            """)

    # memecah setiap kelas pada kolom "kpr_aktif" menjadi kolom tersendiri
    split_kolom_kpr = pd.get_dummies(X["kpr_aktif"], prefix="KPR")
    X = X.join(split_kolom_kpr)

    # menghapus kolom kpr_aktif dari tabel
    X = X.drop(columns = "kpr_aktif")

    st.write("Dataframe rata-rata overdue, risk rating dan kpr aktif sudah di hapus dari data")
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

with modeling:
    st.write("# Modeling")
    naivebayes,decisiontree,k_n_n = st.tabs(["Gaussian Naive Bayes", "Decision Tree", "KNN"])

    with naivebayes:

        gnb = joblib.load('gnb.pkl')
        Y_pred_gnb = gnb.predict(X_test)
        accuracy_gnb = round(accuracy_score(y_test,Y_pred_gnb)* 100, 2)
        label_gnb = pd.DataFrame(
                data={"Label Test" : y_test, "Label Predict" : Y_pred_gnb})
        st.success(f"Akurasi terhadap data test = {accuracy_gnb}")
        st.dataframe(label_gnb)

    with decisiontree:

        dtr = joblib.load('dtr.pkl')
        Y_pred_dtr= dtr.predict(X_test)
        accuracy_tree = round(accuracy_score(y_test,Y_pred_dtr)* 100, 2) 
        label_tree = pd.DataFrame(
                data={"Label Test" : y_test, "Label Predict" : Y_pred_dtr})
        st.success(f"Akurasi terhadap data test = {accuracy_tree}")
        st.dataframe(label_tree)

    with k_n_n:

        knn1 = joblib.load('knn1.pkl')
        Y_pred_knn = knn1.predict(X_test)
        accuracy_knn = round(accuracy_score(y_test,Y_pred_knn)* 100, 2)
        label_knn = pd.DataFrame(
                data={"Label Test" : y_test, "Label Predict" : Y_pred_knn})
        st.success(f"Akurasi terhadap data test = {accuracy_knn}")
        st.dataframe(label_knn)

FIRST_IDX=0

with implementasi:
    indeks_pertama = 0
    
    st.write("# Implementation")
    nama_nasabah = st.text_input('Masukkan Nama Nasabah')
    jumlah_gaji = st.number_input('Masukkan pendapatan pertahun')
    kpr = st.radio("KPR", ("aktif", "tidak aktif"))
    durasi = st.number_input('Masukkan Durasi Peminjaman')
    tanggungan = st.number_input('Masukkan Jumlah Tanggungan')
    overdue = st.selectbox("Overdue", ("0 - 30 days", "31 - 45 days", "46 - 60 days", "61 - 90 days", "> 90 days"))

    kpr_ya = 0
    kpr_tidak = 0
    if kpr == "aktif":
        kpr_ya = 1
    else:
        kpr_tidak = 1
        
    overdue_0=0
    overdue_31=0
    overdue_46=0
    overdue_61=0
    overdue_91=0
    if(overdue == "0 - 30 days"):
        overdues_0 = 1
    elif(overdue == "31 - 45 days"):
        overdues_31 = 1
    elif(overdue == "46 - 60 days"):
        overdues_46 = 1
    elif(overdue == "61 - 90 days"):
        overdues_61 = 1
    else:
        overdues_91 = 1

    cek_gnb = st.button('Cek Risk Ratio dengan Naive Bayes')
    cek_tree = st.button('Cek Risk Ratio dengan Decision Tree')
    cek_knn = st.button('Cek Risk Ratio dengan KNN')

    if cek_gnb:
        result_gnb = gnb.predict([[overdue_0,overdue_31,overdue_46,overdue_61,overdue_91, kpr_tidak,kpr_ya,jumlah_gaji,durasi,tanggungan]])[indeks_pertama]
        st.write(f"Customer Name : ", nama_nasabah,  "has risk rating", result_gnb ,"based on Gaussian Naive Bayes model")
    
    if cek_tree:
        result_tree = dtr.predict([[overdue_0,overdue_31,overdue_46,overdue_61,overdue_91, kpr_tidak,kpr_ya,jumlah_gaji,durasi,tanggungan]])[indeks_pertama]
        st.write(f"Customer Name : ", nama_nasabah,  "has risk rating", result_tree ,"based on Decision Tree model")

    if cek_knn:
        result_knn = knn1.predict([[overdue_0,overdue_31,overdue_46,overdue_61,overdue_91, kpr_tidak,kpr_ya,jumlah_gaji,durasi,tanggungan]])[indeks_pertama]
        st.write(f"Customer Name : ", nama_nasabah,  "has risk rating", result_knn ,"based on KNN model")
