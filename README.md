# Face Emotion Detection With OepenCV & Random Forest
Code ini adalah program Capture Image OpenCV yang dapat mengklasifikasi ekspresi wajah secara realtime dengan menggunakan webcam dari pc/laptop. Program ini menggunakan algoritma machine learning Random Forest yang telah dilatih dengan dataset FER2013 yang didapatkan dari Kaggle.


# Cara Menjalankan Code
## 1. Install library yang diperlukan
   - OpenCV `pip install opencv-python`

   + Numpy `pip install numpy`

   * Scikit-Image `pip install scikit-image`

   - Joblib `pip install joblib`

Jalankan Command diatas pada cmd/powershell

## 2. Jalankan Code utama
setelah semua library berhasil di install, buka file code dengan nama `AI_Visual.py` menggunakan VSCode lalu klik tombol run.

## 3. Penggunaan
Setelah file berhasil dijalankan, akan muncul windows yang akan langsung membuka kamera pada device anda. Usahakan posisi wajah sesuai supaya dapat terdeteksi oleh sistem. Setelah wajah berhasil terdeteksi, sistem akan secara otomatis mengklasifikasikan ekspresi wajah anda.
