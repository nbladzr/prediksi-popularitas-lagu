# Prediksi Popularitas Lagu 🎶

Proyek ini merupakan aplikasi Machine Learning yang bertujuan untuk memprediksi popularitas lagu berdasarkan fitur-fitur audio yang tersedia dalam dataset Spotify, seperti:
- Danceability
- Energy
- Loudness
- Speechiness
- Acousticness
- Instrumentalness
- Liveness
- Valence
- Tempo

Aplikasi ini dibangun menggunakan framework **Streamlit**, sehingga dapat berjalan langsung di browser tanpa instalasi rumit. 

## Fitur Utama
- **Prediksi melalui CSV**  
  Pengguna dapat mengunggah file CSV berisi kumpulan data lagu dan aplikasi akan memberikan prediksi popularitas secara otomatis.

- **Prediksi Input Manual**  
  Pengguna dapat memasukkan fitur-fitur lagu secara manual melalui antarmuka slider dan input box.

- **Download hasil prediksi**  
  Output prediksi dapat diunduh dalam format CSV.

Model Machine Learning dilatih menggunakan library **scikit-learn** dan disimpan sebagai file `model.pkl`.

Aplikasi ini sangat cocok untuk pembelajaran Machine Learning, analisis data musik, serta pembuatan prototipe sistem prediksi.
