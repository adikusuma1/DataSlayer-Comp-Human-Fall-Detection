# 🧠 Human Fall Detection - DataSlayer ML Competition

## 📌 Overview
Human Fall Detection adalah sistem berbasis visi komputer untuk mengidentifikasi apakah seseorang sedang terjatuh pada citra. Proyek ini dikembangkan untuk kompetisi **DataSlayer ML** dengan target klasifikasi biner: jatuh vs. tidak jatuh.

### 🎯 Goals
- Mendeteksi kejadian jatuh dari gambar statis
- Optimasi model untuk akurasi dan generalisasi tinggi
- Berpartisipasi dalam competitive

---

## 🏆 Highlights
- ✅ **Top 10%** pada leaderboard (Kaggle Score: **0.91**)
- ✅ **Validation Accuracy:** 99.3%
- ✅ **Validation AUC:** 99.9%
- ✅ Dilengkapi visualisasi: distribusi kelas, metrik pelatihan, dan prediksi

---
## 📁 Project Structure
📦 fall-detection-project
├── chart/                            # Output visualisasi model
│   ├── output.png                    # Visualisasi prediksi akhir
│   ├── training_metrics.png         # Grafik akurasi & loss per epoch
│   └── persebaran data masing masing kelas.png  # Distribusi label
├── hyperparameter_tuning/           # File & log tuning hyperparameter
├── train/                           # Folder dataset latih
├── test/                            # Folder dataset uji (tanpa label)
├── best_fall_detection_model.h5     # Model terbaik berdasarkan validasi
├── final_fall_detection_model.h5    # Model final setelah tuning
├── main.py                          # Skrip utama pelatihan & prediksi
├── sample_submission.csv            # Template pengumpulan prediksi
├── submission_duelist_seannamon.csv # Hasil akhir untuk kompetisi
└── README.md                        # Dokumentasi proyek (file ini)

---

## ⚙️ Technologies Used
- 🐍 Python 3.10.11
- 🔶 TensorFlow & Keras (untuk deep learning)
- 🔍 Keras Tuner (untuk hyperparameter search)
- 📊 Matplotlib, Seaborn (untuk visualisasi)
- 📦 OpenCV (untuk pemrosesan gambar)
- 📄 Pandas, NumPy (untuk manipulasi data)

🧠 Model Architecture
- 📚 Base model: ResNet50 pretrained dari ImageNet
- ❄️ Layer freezing: hanya melatih top layer
- 🧠 Custom classifier head:
- GlobalAveragePooling2D
- Dropout
- Dense (Softmax untuk klasifikasi biner)

⚙️ Optimizations:
- Adaptive learning rate
- Dropout tuning
- Class weight balancing
- EarlyStopping untuk mencegah overfitting
